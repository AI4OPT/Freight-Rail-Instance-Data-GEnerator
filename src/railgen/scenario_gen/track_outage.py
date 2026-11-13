from itertools import count

from pydantic import BaseModel, Field

from railgen.models import DayOfWeek, YdKey
from railgen.scenario_gen.time import SimulationMinute, TimeBound, WeekMinute
from railgen.utils import MINUTES_PER_DAY, MINUTES_PER_WEEK


class TrackOutage(BaseModel):
    """Used to model a temporarily unavailable track segment connecting two yards.
    This can be used to simulate events like track maintenance, weather events, bridge collapses, etc.
    """

    yards: tuple[YdKey, YdKey] = Field(
        description="Pair of yards between which the track outage occurs. The outage is applied in both directions."
    )
    time_bounds: list[TimeBound] = Field(
        description="List of time ranges during which the outage occurs.", min_length=1
    )


def determine_outage_delay(
    origin: YdKey,
    dest: YdKey,
    outages: list[TrackOutage],
    *,
    sim_minute: SimulationMinute,
    day: DayOfWeek,
    travel_minutes: float,
) -> None | int:
    """If the travel track has any delays during the travel time, this returns the total resulting delay in minutes."""

    # Grab outages that apply to this specific track segment
    track_outages = [
        outage
        for outage in outages
        if (origin, dest) == outage.yards or (dest, origin) == outage.yards
    ]
    if len(track_outages) == 0:
        return None

    # convert sim time into week minute
    week_start = (day.to_day_int() - 1) * MINUTES_PER_DAY
    week_minute = week_start + sim_minute.min_int()
    week_minute = WeekMinute(week_minute % MINUTES_PER_WEEK)

    # convert all of the outages into non-overlapping, SORTED time windows
    outage_windows = _merge_outage_windows(track_outages)

    # This creates a lazy(!) generator that keeps yielding the windows
    # wrapping around each week. This allows us to handle cases
    # where a train is delayed into the next week where we still
    # need to account for track outages.
    weekly_windows = (
        (start + week, end + week)
        for week in count(0, MINUTES_PER_WEEK)
        for start, end in outage_windows
    )
    delay = 0
    for start, end in weekly_windows:
        if start >= week_minute + travel_minutes + delay:
            break
        if end > week_minute:
            delay += end - max(start, week_minute)
    return delay


def _merge_outage_windows(
    track_outages: list[TrackOutage],
) -> list[tuple[WeekMinute, WeekMinute]]:
    """Returns a list of non-overlapping, sorted outage windows for the given track outages."""
    assert len(track_outages) > 0

    outage_windows: list[tuple[WeekMinute, WeekMinute]] = []
    for outage in track_outages:
        for time_bound in outage.time_bounds:
            start_minute, end_minute = time_bound.as_minutes()
            outage_windows.append((start_minute, end_minute))

    # sort by window start time
    outage_windows.sort(key=lambda win: win[0])

    # merge overlapping windows
    merged: list[tuple[WeekMinute, WeekMinute]] = []
    cur_start = outage_windows[0][0]
    cur_end = outage_windows[0][1]
    # walk the remaining windows and merge
    for start, end in outage_windows[1:]:
        if start > cur_end:
            merged.append((cur_start, cur_end))
            cur_start = start
            cur_end = end
        else:
            cur_end = max(cur_end, end)
    merged.append((cur_start, cur_end))

    return merged
