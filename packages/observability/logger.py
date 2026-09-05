"""
Structured observability — every analysis gets request_id, run_id, timings, provenance.
Separates INFO/WARN/ERROR; never exposes stack traces to users.
"""
import logging
import json
import time
import uuid
from typing import Dict, Any, Optional

# Configure root
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("satquery")

def get_logger(name: str = "satquery") -> logging.Logger:
    return logging.getLogger(name)

def structured_log(level: str, event: str, **kwargs):
    payload = {"event": event, **kwargs, "ts": time.time()}
    msg = json.dumps(payload, default=str)
    if level == "error":
        logger.error(msg)
    elif level == "warning":
        logger.warning(msg)
    else:
        logger.info(msg)

def new_request_id() -> str:
    return str(uuid.uuid4())[:8]

def new_run_id() -> str:
    return str(uuid.uuid4())
