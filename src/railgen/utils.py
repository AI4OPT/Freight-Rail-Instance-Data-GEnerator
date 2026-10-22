from datetime import UTC, datetime
from typing import Annotated

MINUTES_PER_HOUR = 60

MINUTES_PER_DAY = 24 * 60
MINUTES_PER_WEEK = 7 * 24 * 60


HourMinute = Annotated[str, "Format: HH:MM"]


def get_elapsed_day(elapsed: int, start: int) -> int:
    """Return number of days relative to start of the trip."""
    # Calculate `time_remaining_first_day` to the start of the trip to prevent wrong day reporting at end of the first segment
    time_remaining_first_day = MINUTES_PER_DAY - (start % MINUTES_PER_DAY)
    if elapsed < time_remaining_first_day:
        return 1
    return (elapsed - time_remaining_first_day) // MINUTES_PER_DAY + 2


def get_abs_HHMM(minute: int) -> HourMinute:
    """Convert absolute minutes to "HH:MM")"""
    remaining_minutes = minute % MINUTES_PER_DAY

    hour = remaining_minutes // MINUTES_PER_HOUR
    min = remaining_minutes % MINUTES_PER_HOUR

    hour_str = f"{int(hour):02d}"
    min_str = f"{int(min):02d}"

    return f"{hour_str}:{min_str}"


def get_min_from_day_and_HHMM(elapsed_day_str: str, hhmm: HourMinute) -> int:
    """Convert day and "HH:MM" to absolute minutes.

    Examples:
    - day 1, "00:00" = 0 minutes
    - day 2, "00:00" = 1440 minutes
    - day 3, "12:30" = 2880 + 750 = 3630 minutes
    """
    try:
        day = int(elapsed_day_str)
    except ValueError:
        raise ValueError(f"Invalid elapsed_day value: {elapsed_day_str}")
    assert day >= 1, "day must be >= 1"

    parts = hhmm.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid HH:MM format: {hhmm}")

    try:
        hour = int(parts[0])
    except ValueError:
        raise ValueError(f"Invalid hour value in HH:MM format: {hhmm}")
    try:
        minute = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid minute value in HH:MM format: {hhmm}")

    return minute + (hour * MINUTES_PER_HOUR) + ((day - 1) * MINUTES_PER_DAY)


def kilometers_to_meters(kilometers: float) -> float:
    return kilometers * 1000


def get_progress_desc_with_padding(description: str) -> str:
    """Returns description string for progress bars with padding"""
    DESCRIPTION_LENGTH = 35
    padding = " " * (DESCRIPTION_LENGTH - len(description))
    return description + padding


def directory_timestamp() -> str:
    """Timestamp formatted for use in output directory names."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%Hh%Mm%SsZ")
