"""Retraining monitoring package.

The app is usually started from ``backend_connector`` while this package
lives one folder higher, so make the connector modules importable here too.
"""

import sys
from pathlib import Path

BACKEND_CONNECTOR_DIR = Path(__file__).resolve().parents[1] / "backend_connector"
if str(BACKEND_CONNECTOR_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_CONNECTOR_DIR))
