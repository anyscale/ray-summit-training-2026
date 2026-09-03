"""Shared modules for the course notebooks.

Imported as a package so every notebook and every Ray worker resolves them the
same way -- Ray puts the runtime_env working_dir (this repo root) on sys.path,
so `from tools import util` works on the driver and on the workers alike.

    from tools import cluster, util
    from tools.lerobot_datasource import LeRobotDatasource
    from tools.policy_server import PI05PolicyServer

`tools/sim_worker.py` is the exception: it runs as a standalone subprocess
(`python -u tools/sim_worker.py`), so it imports its sibling franka_env directly.
"""
