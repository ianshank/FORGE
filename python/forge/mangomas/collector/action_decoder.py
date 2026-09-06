"""Action name decoding for FORGE discrete action spaces."""

from __future__ import annotations

_FORGE_BASE_ACTIONS = 40
_FORGE_DRONE_ACTION_COUNT = 19
_FORGE_AGRI_ACTION_COUNT = 14
_FORGE_HEX_ACTION_COUNT = 6


def _decode_non_drone_action_name(action_id: int, comm_vocab_size: int) -> str | None:
    action_label: str | None = None
    if action_id == 0:
        action_label = "Noop"
    elif 1 <= action_id <= 4:
        action_label = ("MoveUp", "MoveDown", "MoveLeft", "MoveRight")[action_id - 1]
    elif action_id == 5:
        action_label = "PickUp"
    elif 6 <= action_id <= 15:
        action_label = "Drop"
    elif 16 <= action_id <= 25:
        action_label = "Use"
    elif 26 <= action_id <= 34:
        action_label = "Craft"
    elif 35 <= action_id <= 38:
        action_label = "Push"
    elif action_id == 39:
        action_label = "Interact"
    elif _FORGE_BASE_ACTIONS <= action_id < _FORGE_BASE_ACTIONS + comm_vocab_size:
        action_label = "Communicate"
    return action_label


def _decode_drone_action_name(drone_offset: int) -> str | None:
    if drone_offset <= 4:
        return ("Ascend", "Descend", "Hover", "TakeOff", "Land")[drone_offset]
    if 5 <= drone_offset <= 8:
        return "Scan"
    if 9 <= drone_offset <= 18:
        return "DropPayload"
    return None


def _decode_agri_action_name(agri_offset: int) -> str | None:
    if 0 <= agri_offset <= 9:
        return "Spray"
    if agri_offset == 10:
        return "ScanMultispectral"
    if agri_offset == 11:
        return "ScanThermal"
    if agri_offset == 12:
        return "RelaySoilData"
    if agri_offset == 13:
        return "GenerateReport"
    return None


def _decode_hex_action_name(hex_offset: int) -> str | None:
    if 0 <= hex_offset < _FORGE_HEX_ACTION_COUNT:
        return "Move"
    return None


def decode_action_name(
    action_id: int,
    comm_vocab_size: int,
    drone_enabled: bool,
    agri_enabled: bool = False,
    hex_enabled: bool = False,
) -> str:
    """Decode a FORGE discrete action id into a stable semantic label."""
    non_drone_label = _decode_non_drone_action_name(action_id, comm_vocab_size)
    if non_drone_label is not None:
        return non_drone_label

    offset = action_id - _FORGE_BASE_ACTIONS - comm_vocab_size
    if offset < 0:
        return "UnknownAction"

    if drone_enabled:
        drone_label = _decode_drone_action_name(offset)
        if drone_label is not None:
            return drone_label
        offset -= _FORGE_DRONE_ACTION_COUNT

    if agri_enabled and drone_enabled:
        agri_label = _decode_agri_action_name(offset)
        if agri_label is not None:
            return agri_label
        offset -= _FORGE_AGRI_ACTION_COUNT

    if hex_enabled:
        hex_label = _decode_hex_action_name(offset)
        if hex_label is not None:
            return hex_label

    return "UnknownAction"
