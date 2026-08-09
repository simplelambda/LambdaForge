"""Actions available to the adaptive experiment controller."""

from enum import Enum


class AdaptiveActionKind(str, Enum):
    """Name one scientifically distinct controller decision."""

    START_NEW = "start_new"
    RESUME = "resume"
    ADD_SEED = "add_seed"
    PAUSE = "pause"
    DROP = "drop"
    CONFIRM = "confirm"
