# Data

The raw data are **not** in this repository (see "Why raw data is not committed").
`data/raw/*` and `data/processed/*` are git-ignored.

## Source

Monash Smart Infrastructure (MSI) dataset, folder
**"MMA3001 Dataset - MSI Occupancy and Environmental"** on the unit's Google
Drive. Download:

| File | Content |
|---|---|
| `5EnvSensor_MayToDec2024_180kRows.zip` | environmental sensor reports (~180 k rows) |
| `5occupancySensor_MayToDec2024_9MRows.zip` | occupancy-counter events (~9 M rows) |
| `Sensor ID and Locations.xlsx` | sensor IDs, buildings/rooms (sheets `ENV` and `OCC`) |

The data are restricted to the unit, so they are not redistributed here;
request them from the MMA3001 teaching team.

## Where to put it

Unzip so the layout is exactly:

```
data/raw/
  env/5EnvSensor_MayToDec2024_180kRows.csv        # any single *.csv in env/
  occ/5occupancySensor_MayToDec2024_9MRows.csv    # any single *.csv in occ/
  Sensor ID and Locations.xlsx
```

`scripts/ingest.py` reads the first `*.csv` found in `data/raw/env/` and
`data/raw/occ/`. Then run `make ingest`.

## Raw schemas

**Environmental sensors** (`env/*.csv`, one row per sensor report):

| Column | Meaning |
|---|---|
| `id` | row id |
| `sensorid` | sensor identifier (e.g. `6012002000869`) |
| `devicetype` | device model |
| `jsondata` | JSON list of `{"values": [{"time": <unix s>, "value": x}], "variable": {"name", "unit"}}` objects |
| `status`, `processing_errors` | ingestion status / errors |
| `createdate`, `processdate` | database timestamps (not used; the measurement time inside `jsondata` is) |

Variables used (name -> unit): Carbon dioxide -> ppm; Temperature -> degC;
Humidity -> % relative humidity; Pressure -> mb. Also parsed but not used by the
models: Light VOC (ppb), Formaldehyde (ug/m^3), Battery (V). Reports arrive
irregularly (~10 min) and many rows are exact re-sent duplicates.

**Occupancy sensors** (`occ/*.csv`, one row per event per floor-space):

| Column | Meaning |
|---|---|
| `id` | row id |
| `deviceid` | counter device |
| `floorspaceid` | floor-space UUID (the unit of aggregation) |
| `occupancystatus` | `CurrentlyOccupied` / `RecentlyOccupied` / `NotOccupied` |
| `headcount` | people counted [persons] |
| `collecteddate` | local timestamp with UTC offset, e.g. `2024-05-01 09:00:00+10` (+10/+11 across DST) |
| `occupancystatuschangedate`, `previousoccupancystatus` | status-change bookkeeping (not used) |

All timestamps are converted to `Australia/Melbourne`. Despite the file names
("MayToDec2024"), the env timestamps in the audit span 2023-05 to 2026-04
(`reports/results/ingest_log.txt`); the overlap used for modelling is set by
the paired CO2/occupancy record of each sensor/space pair.

## Processed artefacts (written by `scripts/ingest.py`)

| File | Content |
|---|---|
| `data/processed/env.parquet` | tidy wide env table: `sensorid`, `time`, `co2_ppm`, `temp_c`, `rh_pct`, `pressure_mb`, `lvoc_ppb`, `hcho_ugm3`, `battery_v` |
| `data/processed/occ_1min.parquet` | per-floor-space 1-minute time-weighted mean headcount (one column per space) |
| `data/processed/occ_15min.parquet` | 15-minute occupancy aggregate (cache from an earlier pipeline stage; not read by the current scripts) |
| `data/processed/rooms.parquet` | 15-minute room tables (`co2_ppm`, `temp_c`, `rh_pct`, `occupancy`, `room`) for sensor/space pairs with a data-inferred association |
| `reports/results/audit/*.csv`, `reports/results/ingest_log.txt` | audit tables and log (row counts, gaps, flagged values, association scores) |

The parquet caches are reused if present; delete them to force a rebuild.

## Why raw data is not committed

* **Size**: the occupancy export is ~9 M rows.
* **Ownership / privacy**: the data belong to Monash University, are shared
  with permission for coursework only, and describe the use of real
  buildings. They are therefore excluded from the repository and from the
  MIT licence (see the root `README.md`).
