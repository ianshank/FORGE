"""Canonical action definitions, constants, and decoders for FORGE.

Self-contained module at `forge.actions` so both `forge.agents` and
`forge.mangomas` can import action layouts, discrete ID offsets, and
decoders without circular package dependencies.
"""

from __future__ import annotations

from typing import Final

# Slot widths — mirrors forge-types constants (NUM_DIRECTIONS, ACTION_*_SLOTS).
ACTION_MOVE_FAMILY_SIZE: Final[int] = 4
ACTION_DROP_SLOTS: Final[int] = 10
ACTION_USE_SLOTS: Final[int] = 10
ACTION_CRAFT_SLOTS: Final[int] = 9
ACTION_PUSH_FAMILY_SIZE: Final[int] = ACTION_MOVE_FAMILY_SIZE
DRONE_FLIGHT_ACTION_COUNT: Final[int] = 5
DRONE_SCAN_DIRECTION_COUNT: Final[int] = ACTION_MOVE_FAMILY_SIZE
DRONE_PAYLOAD_SLOTS: Final[int] = 10
AGRI_SPRAY_SLOTS: Final[int] = 10
AGRI_SPECIALTY_ACTION_COUNT: Final[int] = 4

ACTION_ID_NOOP: Final[int] = 0
ACTION_ID_MOVE_MIN: Final[int] = ACTION_ID_NOOP + 1
ACTION_ID_MOVE_MAX: Final[int] = ACTION_ID_MOVE_MIN + ACTION_MOVE_FAMILY_SIZE - 1
ACTION_ID_PICKUP: Final[int] = ACTION_ID_MOVE_MAX + 1
ACTION_ID_DROP_MIN: Final[int] = ACTION_ID_PICKUP + 1
ACTION_ID_DROP_MAX: Final[int] = ACTION_ID_DROP_MIN + ACTION_DROP_SLOTS - 1
ACTION_ID_USE_MIN: Final[int] = ACTION_ID_DROP_MAX + 1
ACTION_ID_USE_MAX: Final[int] = ACTION_ID_USE_MIN + ACTION_USE_SLOTS - 1
ACTION_ID_CRAFT_MIN: Final[int] = ACTION_ID_USE_MAX + 1
ACTION_ID_CRAFT_MAX: Final[int] = ACTION_ID_CRAFT_MIN + ACTION_CRAFT_SLOTS - 1
ACTION_ID_PUSH_MIN: Final[int] = ACTION_ID_CRAFT_MAX + 1
ACTION_ID_PUSH_MAX: Final[int] = ACTION_ID_PUSH_MIN + ACTION_PUSH_FAMILY_SIZE - 1
ACTION_ID_INTERACT: Final[int] = ACTION_ID_PUSH_MAX + 1

# Mirrors `forge-types` action encoding (base block before comm/drone/agri/hex).
FORGE_BASE_ACTIONS: Final[int] = ACTION_ID_INTERACT + 1
FORGE_DRONE_ACTION_COUNT: Final[int] = (
    DRONE_FLIGHT_ACTION_COUNT + DRONE_SCAN_DIRECTION_COUNT + DRONE_PAYLOAD_SLOTS
)
FORGE_AGRI_ACTION_COUNT: Final[int] = AGRI_SPRAY_SLOTS + AGRI_SPECIALTY_ACTION_COUNT
FORGE_HEX_ACTION_COUNT: Final[int] = 6

DRONE_OFFSET_ASCEND: Final[int] = 0
DRONE_OFFSET_DESCEND: Final[int] = 1
DRONE_OFFSET_HOVER: Final[int] = 2
DRONE_OFFSET_TAKEOFF: Final[int] = 3
DRONE_OFFSET_LAND: Final[int] = 4
DRONE_OFFSET_SCAN_MIN: Final[int] = DRONE_OFFSET_LAND + 1
DRONE_OFFSET_SCAN_MAX: Final[int] = DRONE_OFFSET_SCAN_MIN + DRONE_SCAN_DIRECTION_COUNT - 1
DRONE_OFFSET_PAYLOAD_MIN: Final[int] = DRONE_OFFSET_SCAN_MAX + 1
DRONE_OFFSET_PAYLOAD_MAX: Final[int] = DRONE_OFFSET_PAYLOAD_MIN + DRONE_PAYLOAD_SLOTS - 1

AGRI_OFFSET_SPRAY_MIN: Final[int] = 0
AGRI_OFFSET_SPRAY_MAX: Final[int] = AGRI_SPRAY_SLOTS - 1
AGRI_OFFSET_MULTISPECTRAL: Final[int] = AGRI_OFFSET_SPRAY_MAX + 1
AGRI_OFFSET_THERMAL: Final[int] = AGRI_OFFSET_MULTISPECTRAL + 1
AGRI_OFFSET_SOIL: Final[int] = AGRI_OFFSET_THERMAL + 1
AGRI_OFFSET_REPORT: Final[int] = AGRI_OFFSET_SOIL + 1

_FORGE_BASE_ACTIONS = FORGE_BASE_ACTIONS
_FORGE_DRONE_ACTION_COUNT = FORGE_DRONE_ACTION_COUNT
_FORGE_AGRI_ACTION_COUNT = FORGE_AGRI_ACTION_COUNT
_FORGE_HEX_ACTION_COUNT = FORGE_HEX_ACTION_COUNT

_MOVE_LABELS: Final[tuple[str, ...]] = ("MoveUp", "MoveDown", "MoveLeft", "MoveRight")
_DRONE_FLIGHT_LABELS: Final[tuple[str, ...]] = ("Ascend", "Descend", "Hover", "TakeOff", "Land")

_SKILL_CATEGORY_BY_LABEL: Final[dict[str, str]] = {
    "Noop": "idle",
    "MoveUp": "navigate",
    "MoveDown": "navigate",
    "MoveLeft": "navigate",
    "MoveRight": "navigate",
    "Move": "navigate",
    "PickUp": "gather",
    "Drop": "gather",
    "Use": "craft",
    "Craft": "craft",
    "Push": "combat",
    "Interact": "combat",
    "Communicate": "communicate",
    "Ascend": "aerial",
    "Descend": "aerial",
    "Hover": "aerial",
    "TakeOff": "aerial",
    "Land": "aerial",
    "Scan": "aerial",
    "DropPayload": "aerial",
    "Spray": "agriculture",
    "ScanMultispectral": "agriculture",
    "ScanThermal": "agriculture",
    "RelaySoilData": "agriculture",
    "GenerateReport": "agriculture",
}

UNKNOWN_ACTION_LABEL: Final[str] = "UnknownAction"
UNKNOWN_SKILL_CATEGORY: Final[str] = "explore"


def _decode_non_drone_action_name(action_id: int, comm_vocab_size: int) -> str | None:
    action_label: str | None = None
    if action_id == ACTION_ID_NOOP:
        action_label = "Noop"
    elif ACTION_ID_MOVE_MIN <= action_id <= ACTION_ID_MOVE_MAX:
        action_label = _MOVE_LABELS[action_id - ACTION_ID_MOVE_MIN]
    elif action_id == ACTION_ID_PICKUP:
        action_label = "PickUp"
    elif ACTION_ID_DROP_MIN <= action_id <= ACTION_ID_DROP_MAX:
        action_label = "Drop"
    elif ACTION_ID_USE_MIN <= action_id <= ACTION_ID_USE_MAX:
        action_label = "Use"
    elif ACTION_ID_CRAFT_MIN <= action_id <= ACTION_ID_CRAFT_MAX:
        action_label = "Craft"
    elif ACTION_ID_PUSH_MIN <= action_id <= ACTION_ID_PUSH_MAX:
        action_label = "Push"
    elif action_id == ACTION_ID_INTERACT:
        action_label = "Interact"
    elif FORGE_BASE_ACTIONS <= action_id < FORGE_BASE_ACTIONS + comm_vocab_size:
        action_label = "Communicate"
    return action_label


def _decode_drone_action_name(drone_offset: int) -> str | None:
    if DRONE_OFFSET_ASCEND <= drone_offset <= DRONE_OFFSET_LAND:
        return _DRONE_FLIGHT_LABELS[drone_offset]
    if DRONE_OFFSET_SCAN_MIN <= drone_offset <= DRONE_OFFSET_SCAN_MAX:
        return "Scan"
    if DRONE_OFFSET_PAYLOAD_MIN <= drone_offset <= DRONE_OFFSET_PAYLOAD_MAX:
        return "DropPayload"
    return None


def _decode_agri_action_name(agri_offset: int) -> str | None:
    if AGRI_OFFSET_SPRAY_MIN <= agri_offset <= AGRI_OFFSET_SPRAY_MAX:
        return "Spray"
    if agri_offset == AGRI_OFFSET_MULTISPECTRAL:
        return "ScanMultispectral"
    if agri_offset == AGRI_OFFSET_THERMAL:
        return "ScanThermal"
    if agri_offset == AGRI_OFFSET_SOIL:
        return "RelaySoilData"
    if agri_offset == AGRI_OFFSET_REPORT:
        return "GenerateReport"
    return None


def _decode_hex_action_name(hex_offset: int) -> str | None:
    if 0 <= hex_offset < FORGE_HEX_ACTION_COUNT:
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

    offset = action_id - FORGE_BASE_ACTIONS - comm_vocab_size
    if offset < 0:
        return UNKNOWN_ACTION_LABEL

    if drone_enabled:
        drone_label = _decode_drone_action_name(offset)
        if drone_label is not None:
            return drone_label
        offset -= FORGE_DRONE_ACTION_COUNT

    if agri_enabled and drone_enabled:
        agri_label = _decode_agri_action_name(offset)
        if agri_label is not None:
            return agri_label
        offset -= FORGE_AGRI_ACTION_COUNT

    if hex_enabled:
        hex_label = _decode_hex_action_name(offset)
        if hex_label is not None:
            return hex_label

    return UNKNOWN_ACTION_LABEL


def skill_category_for_action_name(action_name: str) -> str:
    """Map a decoded action label onto a reusable skill family."""
    return _SKILL_CATEGORY_BY_LABEL.get(action_name, UNKNOWN_SKILL_CATEGORY)


def skill_category_for_action_id(
    action_id: int,
    comm_vocab_size: int,
    drone_enabled: bool,
    agri_enabled: bool = False,
    hex_enabled: bool = False,
) -> str:
    """Decode an action id then map it onto a reusable skill family."""
    label = decode_action_name(
        action_id,
        comm_vocab_size=comm_vocab_size,
        drone_enabled=drone_enabled,
        agri_enabled=agri_enabled,
        hex_enabled=hex_enabled,
    )
    return skill_category_for_action_name(label)


__all__ = [
    "ACTION_CRAFT_SLOTS",
    "ACTION_DROP_SLOTS",
    "ACTION_ID_CRAFT_MAX",
    "ACTION_ID_CRAFT_MIN",
    "ACTION_ID_DROP_MAX",
    "ACTION_ID_DROP_MIN",
    "ACTION_ID_INTERACT",
    "ACTION_ID_MOVE_MAX",
    "ACTION_ID_MOVE_MIN",
    "ACTION_ID_NOOP",
    "ACTION_ID_PICKUP",
    "ACTION_ID_PUSH_MAX",
    "ACTION_ID_PUSH_MIN",
    "ACTION_ID_USE_MAX",
    "ACTION_ID_USE_MIN",
    "ACTION_MOVE_FAMILY_SIZE",
    "ACTION_PUSH_FAMILY_SIZE",
    "ACTION_USE_SLOTS",
    "AGRI_OFFSET_MULTISPECTRAL",
    "AGRI_OFFSET_REPORT",
    "AGRI_OFFSET_SOIL",
    "AGRI_OFFSET_SPRAY_MAX",
    "AGRI_OFFSET_SPRAY_MIN",
    "AGRI_OFFSET_THERMAL",
    "AGRI_SPECIALTY_ACTION_COUNT",
    "AGRI_SPRAY_SLOTS",
    "DRONE_FLIGHT_ACTION_COUNT",
    "DRONE_OFFSET_ASCEND",
    "DRONE_OFFSET_DESCEND",
    "DRONE_OFFSET_HOVER",
    "DRONE_OFFSET_LAND",
    "DRONE_OFFSET_PAYLOAD_MAX",
    "DRONE_OFFSET_PAYLOAD_MIN",
    "DRONE_OFFSET_SCAN_MAX",
    "DRONE_OFFSET_SCAN_MIN",
    "DRONE_OFFSET_TAKEOFF",
    "DRONE_PAYLOAD_SLOTS",
    "DRONE_SCAN_DIRECTION_COUNT",
    "FORGE_AGRI_ACTION_COUNT",
    "FORGE_BASE_ACTIONS",
    "FORGE_DRONE_ACTION_COUNT",
    "FORGE_HEX_ACTION_COUNT",
    "UNKNOWN_ACTION_LABEL",
    "UNKNOWN_SKILL_CATEGORY",
    "_FORGE_AGRI_ACTION_COUNT",
    "_FORGE_BASE_ACTIONS",
    "_FORGE_DRONE_ACTION_COUNT",
    "_FORGE_HEX_ACTION_COUNT",
    "decode_action_name",
    "skill_category_for_action_id",
    "skill_category_for_action_name",
]
