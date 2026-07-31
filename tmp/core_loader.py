"""
Small loader shim.

`5.App_Preparation_v2.py` cannot be imported with a normal `import` statement
because its filename starts with a digit and contains a dot. This module
loads it dynamically via importlib and re-exports it as `core`, so the rest
of the app can simply do:

    from core_loader import core
    core.load_paper_data()
"""

import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parent / "5.App_Preparation_v2.py"
_MODULE_NAME = "findmylab_core_v2"

_spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
core = importlib.util.module_from_spec(_spec)
sys.modules[_MODULE_NAME] = core
_spec.loader.exec_module(core)
