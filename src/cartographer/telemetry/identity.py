from __future__ import annotations

import logging
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

_ID_FILE = Path.home() / ".cartographer3d" / "telemetry_id"


def get_or_create_anonymous_id() -> str:
    """Return a persistent anonymous UUID for this installation.

    The UUID is stored in ``~/.cartographer3d/telemetry_id``.  If the file
    cannot be read or written (e.g. permission error) a transient UUID is
    returned and a warning is logged.
    """
    try:
        _ID_FILE.parent.mkdir(parents=True, exist_ok=True)
        if _ID_FILE.exists():
            content = _ID_FILE.read_text().strip()
            if content:
                return content
        new_id = str(uuid.uuid4())
        _ = _ID_FILE.write_text(new_id)
        return new_id
    except PermissionError as exc:
        logger.warning(
            "Telemetry: could not read/write anonymous ID (%s); using transient ID.",
            exc,
        )
    except (OSError, ValueError) as exc:
        logger.warning(
            "Telemetry: unexpected error reading anonymous ID (%s); using transient ID.",
            exc,
        )
    return str(uuid.uuid4())
