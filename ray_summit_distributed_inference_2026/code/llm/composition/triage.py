"""The cheap path: answer what you can without touching a GPU.

This is the half of the composition that makes it worth doing. A GPU is the most
expensive resource in the app, so the first question for any request is whether it
needs one at all. Here a lookup table settles the common questions and everything
else escalates.

Nothing in this file knows an LLM exists, which is the point: it is an ordinary
CPU deployment, and `app.py` composes it with a `build_llm_deployment` through the
same `.bind()` seam it would use for any other pair of deployments.
"""

from typing import Dict, Optional

from ray import serve

# Answers the cheap path can settle without ever reaching the GPU.
FAQ: Dict[str, str] = {
    "hours": "Support is staffed 09:00 to 17:00 UTC, Monday through Friday.",
    "pricing": "Pricing is usage based and billed per GPU hour.",
    "status": "Live service status is published on the status page.",
}


class TriageLogic:
    """Pure routing logic: no Ray, no Serve, unit-testable on its own."""

    def __init__(self, faq: Dict[str, str]) -> None:
        self.faq = faq

    def route(self, text: str) -> Optional[str]:
        """Return the FAQ key this text can be answered from, or None to escalate."""
        lowered = text.lower()
        for key in self.faq:
            if key in lowered:
                return key
        return None


@serve.deployment(ray_actor_options={"num_cpus": 1})
class Triage:
    """The deployment wrapper. CPU only, so it scales independently of the GPU."""

    def __init__(self, faq: Dict[str, str]) -> None:
        self.logic = TriageLogic(faq)

    async def route(self, text: str) -> Optional[str]:
        return self.logic.route(text)
