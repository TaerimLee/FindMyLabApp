"""
Loader shim for the v3 app.

Loads `5.App_Preparation_v3.py` (which itself loads v2 and re-exports it,
adding v3-specific presentation functions) and exposes it as `core`.
"""

import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parent.parent / "5.App_Preparation_v3.py"
_MODULE_NAME = "findmylab_core_v3"

_spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
core = importlib.util.module_from_spec(_spec)
sys.modules[_MODULE_NAME] = core
_spec.loader.exec_module(core)
