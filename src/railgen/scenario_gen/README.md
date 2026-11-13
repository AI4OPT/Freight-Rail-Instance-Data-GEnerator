# FRIDGE: Scenario Generation

What `uv run fridge scenario` does. Entry point: `generate_scenarios` in `src/cli/scenario.py`; the simulation is in `scenario_generation.py`.

---

## Overview

A generated schedule runs exactly on time. Scenario generation simulates it many times with random departure delays, run times and dwells; each simulation is one possible realization of the week.

Output (in `<schedule-dir>/scenarios/scenario-<timestamp>/`):

- **`scenario.json`**: for every schedule row, the departure and arrival deltas of each simulation, by day (Step 4).
- **`scenario-generation-config.yaml`**: the settings the run used, including the seed.
- **`scenario-generation-report.txt`**: first-departure and final-arrival delays by train category (Step 4).

---

## Options

Run `uv run fridge scenario --help` to view all scenario generation options.

---

## Step 1: Config and Seed (`scenario_config.py`)

Distributions can be written inline or named once in `distributions_pool`. Example:

```yaml
general_settings:
  scenario_count: 50
distributions_pool:
  congested: { distr_type: normal, mean: 20, stdev: 10 }
train_category_distributions_overrides:
  I:
    initial_departure: { distr_type: normal, mean: 0, stdev: 2 }
yard_dwell_distributions:
  NORRIS-1644: congested
leg_trip_time_percent_diff_distributions:
  NEW BOYLES-943:
    NORRIS-1644:
      { distr_type: asymmetric_laplace, location: 0, scale: 3, skew: 2 }
region_distributions:
  - center: { city: Chicago, state: IL }
    radius_km: 50
    dwell_time: congested
track_outages:
  - yards: [NEW BOYLES-943, NORRIS-1644]
    time_bounds:
      - start: { day: tue, hour_min: "08:00" }
        end: { day: tue, hour_min: "14:00" }
```

Every distribution named must exist in `distributions_pool`, and every yard key must have the form `YARDNAME-OBJECTID`. Each train and running day gets its own random generator, seeded from (seed, train symbol, day), so a train's samples do not depend on the other trains. The saved config holds the seed and count actually used.

---

## Step 2: Read the Schedule (`schedule_input.py`)

`schedule.csv` is grouped by train symbol into `ScheduledTrain`s (category, running days, ordered legs). Times become minutes from midnight of the trip's first day. A schedule is rejected if columns are missing, if a symbol's rows disagree on category or running days, or if a leg does not begin where the previous one ended.

---

## Step 3: Simulate Each Train (`scenario_generation.py`)

Each train is simulated `scenario_count` times for each running day. Simulations are independent: nothing carries over between runs, days or trains.

### Sampling

Each quantity is the sum of the train category's sample (`default_distributions`, with `train_category_distributions_overrides`) and, if configured, a place sample:

| Quantity              | Train distribution       | Added place distribution                                             | Unit    |
| --------------------- | ------------------------ | -------------------------------------------------------------------- | ------- |
| First departure delay | `initial_departure`      | `yard_dwell_distributions[origin]`                                   | minutes |
| Run time change       | `trip_time_percent_diff` | `leg_trip_time_percent_diff_distributions[bgn_yd][end_yd]` (one-way) | percent |
| Dwell change          | `dwell_time`             | `yard_dwell_distributions[stop]`                                     | minutes |

A `region_distributions` entry has a `center` (`{lat, lon}`, or `{city, state}` geocoded with Nominatim) and a `radius_km`. Its `dwell_time` applies to every yard within the radius (coordinates from `rail_network/yds_dict.json`, so the network must exist), and its `trip_time_percent_diff` to every leg between two of them. A yard or leg with a distribution from two sources is an error.

Independent samples add means and variances: stdev 5 on a yard plus stdev 5 on the category gives about 7.

### Time Bounds

Any distribution, inline or in `distributions_pool`, can take `time_bounds`: a list of windows of the week in which it is sampled. Outside all of them it adds 0. For example, extra dwell at one yard during the day on Monday and Friday:

```yaml
yard_dwell_distributions:
  NORRIS-1644:
    distr_type: normal
    mean: 30
    stdev: 10
    time_bounds:
      - start: { day: mon, hour_min: "06:00" }
        end: { day: mon, hour_min: "18:00" }
      - start: { day: fri, hour_min: "06:00" }
        end: { day: fri, hour_min: "18:00" }
```

- `day` is `mon`, `tue`, `wed`, `thu`, `fri`, `sat` or `sun`. Full names and `m`, `tu`, `w`, `th`, `f`, `sa`, `su` also work. `hour_min` is 24-hour `HH:MM`.
- In each window `start` is included and `end` is not. Both fall within one week, Monday 00:00 to Sunday 23:59, and `start` must come before `end`. A window cannot cross Sunday midnight. A trip that runs past Sunday midnight is checked against the same windows in the next week. Windows may overlap.
- The time checked is the train's day of week and time of day at the event: the scheduled departure for the first departure delay, the actual departure for a leg's run time change, and the actual arrival at the stop for a dwell change.
- The place sample adds to the train sample. A bounded yard, leg or region distribution adds its delay on top of the category's only during its windows. Outside its windows, a bounded category distribution gives 0 and only the place sample remains.
- A mixture takes one `time_bounds` list for all its components. A component cannot have its own.

### Track Outages

Users may define _track outages_ in the scenario configuration file that close a connection between two yard for some period of time. This can be used to model maintenance periods, weather events, bridge closures, or other similar events. The example config in [Step 1](#step-1-config-and-seed-scenario_configpy) closes the track between NEW BOYLES-943 and NORRIS-1644 on Tuesday from 08:00 to 14:00.

Things to note:

- An outage applies in both directions and only to legs that begin at one of the two yards and end at the other. Legs that pass through the closed section without starting and ending at those yards are not affected.
- If a train is already travelling on a given track when an outage begins, it waits on place until the outage ends and continues its journey. The resultant travel delay is added to the segment's arrival delay.
- The wait can push a train into later outages on the same leg, and it waits for those too. Overlapping windows, from one entry or several, count once. Windows repeat every week, so a leg that runs past Sunday midnight meets the next week's outages.

### Walking the Route

A clock starts at the first scheduled departure.

1. **First departure:** add the departure delay.
2. **Each leg:** add `planned_run_time × (1 + run_time_change / 100)`, plus any time stopped for track outages; this gives the actual arrival.
3. **Each intermediate stop:** the planned dwell is a minimum dwell (`general_settings.minimum_dwell_share` × planned, the work at the stop) plus a buffer (the rest). The actual dwell is `max(minimum_dwell, max(minimum_dwell, planned_dwell − max(arrival_delay, 0)) + extra_dwell)`:
   - a late train uses the buffer to recover, all of its delay if the buffer is large enough;
   - an early train is not held and dwells the planned time;
   - `extra_dwell` is the dwell change at the stop: positive is new delay (work, crew, congestion), negative a quicker stop; the dwell never goes below the minimum dwell.

The deltas `actual − scheduled` of every departure and arrival are recorded, truncated to whole minutes. Delay carries down the route and its spread grows leg by leg; a late train recovers only through dwell buffers or a negative run time change. This is the usual delay propagation rule of timetable analysis (Berger et al., 2011; Goverde, 2010). A simulated trip longer than a week is kept, with a warning.

---

## Step 4: Write Output (`src/cli/scenario.py`, `scenario_report.py`)

`scenario.json`, keyed by schedule `ROW_ID`, with days the train does not run left out:

```json
{
  "ATRN_1_NEW BOYLES-1049_NORRIS-1750": {
    "mon": [
      { "dep_min_delta": 3, "arvl_min_delta": 3 },
      { "dep_min_delta": -1, "arvl_min_delta": -2 }
    ],
    "tue": [...]
  }
}
```

Each day's list has one entry per simulation; entry `i` of every row of a train and day comes from the same simulation.

`scenario-generation-report.txt` (`write_report`), per category and overall, for each train run (one simulation of one train on one day; negative is early): the first-departure and final-arrival delay (mean, 10th, 50th, 90th and 99th percentiles, min, max), the shares of early departures and arrivals, the shares of final arrivals more than 0, 60, 120 and 240 minutes late, and the final-arrival delay by number of legs. It also states the minimum dwell share used.

---

## Distribution Types (`distributions.py`)

| `distr_type`         | Parameters                                                               | Sampled as                                                              |
| -------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| `normal`             | `mean`, `stdev`                                                          | NumPy `Generator.normal`                                                |
| `asymmetric_laplace` | `location`, `scale`, `skew`                                              | `scipy.stats.laplace_asymmetric.rvs`                                    |
| `weibull`            | `shape`, `scale`, `location` (default 0)                                 | `location` + `scale` × NumPy `Generator.weibull(shape)`                 |
| `log_normal`         | `log_mean`, `log_stdev`, `location` (default 0)                          | `location` + NumPy `Generator.lognormal(log_mean, log_stdev)`           |
| `mixture`            | `components`: list of `weight`, `negate` (default false), `distribution` | one component drawn with probability `weight`; sign flipped if `negate` |

- **Asymmetric Laplace:** a sharp peak and a longer tail on one side, as train delays have. `skew` is SciPy's `kappa`: below 1 the longer tail is late, above 1 early, 1 is symmetric. It is two exponentials around `location`: late with probability 1 / (1 + `skew`²) and mean `scale` / `skew`, early otherwise with mean `scale` × `skew`.
- **Weibull and lognormal:** nothing below `location` and a long tail above it, the shapes most often fitted to late train delays. Weibull median: `location` + `scale` × (ln 2)^(1/`shape`); `shape` below 1 gives a heavier tail, and `shape: 1` is the exponential with mean `scale`, the most common model of primary train delays (Goverde et al., 2001). Lognormal: `log_mean`, `log_stdev` describe ln(value − `location`), median `location` + e^`log_mean`. A negative `location` allows a train to be early by at most that much; e.g. `{distr_type: weibull, shape: 0.8, scale: 300, location: -30}` gives at least −30 min, median about 160 min.
- **Code defaults** (keys a config leaves out), the same as `configs/scenario_config.yaml`. They are assumptions, not fits to records:
  - `initial_departure`: `mixture`, 40% early (exponential, mean 86.6 min) and 60% late (`weibull`, `shape: 0.5`, `scale: 30`, median about 14 min).
  - `trip_time_percent_diff`: `log_normal`, `location: -20`, `log_mean: 3`, `log_stdev: 0.32`: median about 0%, never more than 20% faster than planned.
  - `dwell_time`: `log_normal`, `location: -40`, `log_mean: 3.85`, `log_stdev: 0.4`: median about +7 min, about a third of the samples negative; the dwell never goes below the minimum dwell.
- **Mixture:** separate early and late sides. Weights are the shares of each side and add up to 1; a component cannot be a mixture. With `negate: true` the early side is given as positive minutes early.

References:

- Berger, A., Gebhardt, A., Müller-Hannemann, M. and Ostrowski, M. (2011). Stochastic delay prediction in large train networks. _ATMOS 2011_, OASIcs 20, 100–111.
- Goverde, R. M. P., Hansen, I. A., Hooghiemstra, G. and Lopuhaä, H. P. (2001). Delay distributions in railway stations. _Proceedings of the 9th World Conference on Transport Research_, Seoul.
- Goverde, R. M. P. (2010). A delay propagation algorithm for large-scale railway traffic networks. _Transportation Research Part C_, 18(3), 269–287.
