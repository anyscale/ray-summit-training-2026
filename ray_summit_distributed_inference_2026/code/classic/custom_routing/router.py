"""A custom request router: send the same user to the same replica.

The default router (Power of Two Choices) balances on queue length and knows
nothing about what a replica has cached. That is the right default and the wrong
one for a feature-backed model: if each replica caches the user features it has
already fetched, then routing a user to a random replica throws that cache away
and pays the feature-store round trip again.

`UserAffinityRouter` routes on the user id instead, and falls back to load when
affinity would hurt. The shape mirrors what `PrefixCacheAffinityRouter` does for
an LLM's KV cache, applied to an ordinary online model.

The API, all from `ray.serve.request_router`:

  - `choose_replicas` is the one required method. It returns a list of RANKS, not
    a list of replicas: `[[first_choice], [fallbacks...]]`. Serve tries rank 0,
    then rank 1, and so on, so a preference degrades instead of failing.
  - `initialize_state` is where `request_router_kwargs` arrives, NOT `__init__`.
    Serve builds the router with its own keyword arguments and calls this
    afterwards, so a subclass that takes its tuning in `__init__` never receives
    it and silently runs on defaults.
  - `select_available_replicas()` filters the candidates down to those with room
    under `max_ongoing_requests`. Call it first; routing to a full replica just
    queues.
  - `on_request_routed` is an optional hook for updating router state after the
    fact. Used here to count what actually landed where.
  - `FIFOMixin` makes the router serve pending requests in arrival order. Without
    it, request ordering is unspecified.
  - `replica.routing_stats` is whatever the deployment's `record_routing_stats()
    returned, polled every `request_routing_stats_period_s`. That is the channel
    a replica uses to tell the router about its own state (see app.py).

This API is alpha, so expect the signatures to move.
"""

import logging
import zlib
from typing import Dict, List, Optional

from ray.serve.request_router import (
    FIFOMixin,
    PendingRequest,
    ReplicaID,
    ReplicaResult,
    RequestRouter,
    RunningReplica,
)

logger = logging.getLogger("ray.serve")


class UserAffinityRouter(FIFOMixin, RequestRouter):
    """Route by user id, with a load-based escape hatch.

    `imbalance_threshold` is the escape hatch: if the affine replica is carrying
    this many more requests than the least-loaded one, affinity loses and the
    request goes to the least-loaded replica instead. Without it, one hot user
    would pin load to a single replica no matter how busy it got.
    """

    #: Set by initialize_state, so it always has a value even if Serve is
    #: constructed without request_router_kwargs.
    imbalance_threshold: int = 10

    def initialize_state(self, imbalance_threshold: int = 10) -> None:
        """Receive `request_router_kwargs`.

        NOT `__init__`. Serve constructs the router itself with its own keyword
        arguments (deployment_id, handle_source, node id, and so on) and then
        calls this with the contents of `request_router_kwargs`. A subclass that
        takes its tuning in `__init__` silently never receives it.

        No `**kwargs` on purpose: a misspelled key in `request_router_kwargs`
        then raises a TypeError naming it, instead of being swallowed and leaving
        the default silently in place.
        """
        self.imbalance_threshold = imbalance_threshold
        # Where each user's requests actually landed, filled in by
        # on_request_routed. A user with more than one replica in its set means
        # affinity was overridden, which is the number worth watching.
        self.replicas_per_user: Dict[str, set] = {}

    def _user_id(self, pending_request: PendingRequest) -> Optional[str]:
        """Pull the user id out of the request, or None if it is not there.

        The router sees the same arguments the deployment will. Anything not
        shaped like a user-keyed request routes on load instead, so an unexpected
        payload degrades rather than raising inside the router.
        """
        for value in list(pending_request.args) + list(pending_request.kwargs.values()):
            user_id = getattr(value, "user_id", None)
            if isinstance(user_id, str):
                return user_id
            if isinstance(value, dict) and isinstance(value.get("user_id"), str):
                return value["user_id"]
        return None

    def _load_of(self, replica: RunningReplica) -> float:
        """Replica load, from the stats the replica reports about itself.

        `record_routing_stats()` in app.py publishes `ongoing_requests`. A replica
        that has not reported yet has an empty dict, and 0 is the right assumption
        for a fresh replica.
        """
        return float(replica.routing_stats.get("ongoing_requests", 0))

    def _affine_replica(
        self, user_id: str, replicas: List[RunningReplica]
    ) -> RunningReplica:
        """Pick this user's replica, deterministically across router restarts.

        CRC32 rather than the built-in hash, which is salted per process and would
        send the same user somewhere different after a restart. Sorting first makes
        the choice independent of the order Serve happens to list replicas in.

        Plain modulo is the simplification here: changing the replica count
        reshuffles every user, not just 1/N of them. That is what a consistent
        hash ring fixes, and `ray.serve.experimental.consistent_hash_router`
        already implements one.
        """
        ordered = sorted(replicas, key=lambda r: str(r.replica_id))
        return ordered[zlib.crc32(user_id.encode("utf-8")) % len(ordered)]

    async def choose_replicas(
        self,
        candidate_replicas: List[RunningReplica],
        pending_request: Optional[PendingRequest] = None,
    ) -> List[List[RunningReplica]]:
        available = self.select_available_replicas(candidate_replicas)
        if not available:
            return []

        user_id = self._user_id(pending_request) if pending_request else None
        if user_id is None:
            return [available]          # nothing to be affine to: one rank, all replicas

        preferred = self._affine_replica(user_id, available)
        least_loaded = min(available, key=self._load_of)

        # Affinity loses when it would pile onto an already-hot replica.
        if self._load_of(preferred) - self._load_of(least_loaded) >= self.imbalance_threshold:
            logger.debug(
                "user=%s dropping affinity: preferred load %.0f vs least %.0f",
                user_id, self._load_of(preferred), self._load_of(least_loaded),
            )
            return [[least_loaded]]

        fallbacks = [r for r in available if r.replica_id != preferred.replica_id]
        return [[preferred], fallbacks] if fallbacks else [[preferred]]

    def on_request_routed(
        self,
        pending_request: PendingRequest,
        replica_id: ReplicaID,
        result: ReplicaResult,
    ) -> None:
        """Record where the request actually went.

        The hook fires after routing, so it sees the outcome rather than the
        intent: a user whose set grows past one replica is a user whose affinity
        the imbalance threshold overrode. That is the signal for whether
        `imbalance_threshold` is tuned too tight.
        """
        user_id = self._user_id(pending_request)
        if user_id is None:
            return
        landed_on = self.replicas_per_user.setdefault(user_id, set())
        landed_on.add(str(replica_id))
        if len(landed_on) > 1:
            logger.debug(
                "user=%s has now used %d replicas: affinity was overridden",
                user_id, len(landed_on),
            )
