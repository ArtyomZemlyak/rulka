"""Map training.algorithm to wiring module (make_network, make_trainer, make_inferer, warmup_compile).

BTR (Beyond The Rainbow) is not registered here: it is the ``btr:`` section of the same YAML, toggling
features inside the IQN stack — always use ``iqn`` wiring when training IQN with or without BTR flags.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Optional

_ALGORITHM_MODULES: dict[str, str] = {
    "iqn": "trackmania_rl.agents.algorithms.iqn_wiring",
    "ppo": "trackmania_rl.agents.algorithms.ppo_wiring",
}


def get_wiring(name: Optional[str] = None) -> ModuleType:
    from config_files.config_loader import get_config

    key = name if name is not None else get_config().algorithm
    mod_path = _ALGORITHM_MODULES.get(key)
    if mod_path is None:
        raise ValueError(
            f"Unknown RL algorithm {key!r}. Supported: {sorted(_ALGORITHM_MODULES)}"
        )
    return importlib.import_module(mod_path)
