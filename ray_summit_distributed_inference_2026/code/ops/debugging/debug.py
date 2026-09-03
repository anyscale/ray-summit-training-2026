"""Debug a Serve application in-process, with no cluster in the way.

The usual debugging problem with Serve is that your code runs inside a replica
actor on some other process, so a breakpoint never trips and a stack trace comes
back as a string. `serve.run(app, _local_testing_mode=True)` removes the actors:
deployments become local objects and a handle call becomes an ordinary function
call, so `pdb`, an IDE breakpoint, and a plain traceback all work again.

What that buys, and what it costs:

  - Bugs it finds. Logic errors, bad shapes, exceptions inside a deployment,
    wrong composition wiring. Anything that is really about your Python.
  - Bugs it hides. Everything about being distributed: serialization failures,
    resource scheduling, replica placement, autoscaling behavior, and the actual
    concurrency. Local mode runs your code in one process, so a race between
    replicas cannot reproduce here.

`_local_testing_mode` is a private parameter. It works (verified against Ray
2.54) and is the documented way to do this, but expect the name to change.

Run:
    python debug.py
"""

import asyncio

import numpy as np
import torch
from ray import serve
from ray.serve.handle import DeploymentHandle


@serve.deployment
class Featurizer:
    """A cheap upstream deployment, so the graph has more than one hop."""

    async def scale(self, features: list[float]) -> list[float]:
        tensor = torch.tensor(features)
        # Unguarded normalization. On an all-zero input the norm is 0, so this
        # divides 0 by 0 and returns NaN without raising anything.
        return (tensor / tensor.norm()).tolist()


@serve.deployment
class Model:
    """The model under test. Nothing here knows it is running locally."""

    def __init__(self, featurizer: DeploymentHandle, in_features: int = 10) -> None:
        self.featurizer = featurizer
        self.model = torch.nn.Linear(in_features, 5)

    async def predict(self, data: dict) -> dict:
        scaled = await self.featurizer.scale.remote(data["features"])
        with torch.no_grad():
            prediction = self.model(torch.tensor(scaled))
        return {"prediction": prediction.tolist()}


app = Model.bind(Featurizer.bind())


async def main() -> None:
    # In local mode this returns a handle backed by plain objects, so the
    # .remote() below is a function call you can step into.
    handle = serve.run(app, _local_testing_mode=True)

    good = {"features": np.random.rand(10).tolist()}
    print("ok:      ", await handle.predict.remote(good))

    # The instructive case. Nothing raises, and the prediction comes back as a
    # list of NaN: the bug is two hops upstream in Featurizer.scale, and the
    # response gives no hint of that. Set a breakpoint in `scale` and run this
    # line again. The debugger stops there, in this process, with a real stack,
    # so you can inspect the tensor that went NaN. Against a live cluster the
    # breakpoint never trips, which is the whole reason local mode exists.
    corrupted = await handle.predict.remote({"features": [0.0] * 10})
    print("all-zero:", corrupted)
    print("all NaN? ", all(np.isnan(v) for v in corrupted["prediction"]))


if __name__ == "__main__":
    asyncio.run(main())
