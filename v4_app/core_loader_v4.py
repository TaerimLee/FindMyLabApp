"""
Loader shim for the v4 app.

Loads `5.App_Preparation_v4.py` (which itself loads v3, which loads v2,
re-exporting each along the way and adding presentation/filtering
functions) and exposes it as `core`.
"""

import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parent.parent / "5.App_Preparation_v4.py"
_MODULE_NAME = "findmylab_core_v4"

_spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
core = importlib.util.module_from_spec(_spec)
sys.modules[_MODULE_NAME] = core
_spec.loader.exec_module(core)
