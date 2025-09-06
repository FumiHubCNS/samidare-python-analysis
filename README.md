# Binary Data Decoder 

Tools to decode binary data, write them to Parquet, and analyze/visualize with PySpark.

This repository is managed with **[uv](https://github.com/astral-sh/uv)**.

## Requirements

- Python (a version supported by uv)
- uv (package manegement tool)
- (When using PySpark) Java + Apache Spark
  - `spark-submit` must be available on your `PATH`

> ⚠️ **Conda conflicts**  
> Before running anything here, **deactivate** Conda (`conda deactivate`).  
> Mixing Conda envs with uv-managed virtualenvs commonly breaks dependency resolution.


## Setup

Let uv resolve dependencies from your `pyproject.toml` / `requirements.txt`.

```bash
uv sync
```

You don’t need to activate a venv manually—prefix your commands with `uv run` and uv will handle it.


## Decoder: binary → Parquet

Main decoder: `src/samidare_util/decoder/binary_dumper_version3.py`  
I/O paths and marker values are controlled by `parameters.toml`.

```bash
# Decode and write Parquet
uv run python src/samidare_util/decoder/binary_dumper_version3.py -d

# Colored binary check dump
uv run python src/samidare_util/decoder/binary_dumper_version3.py -d -b

# Event-by-event preview (matplotlib)
uv run python src/samidare_util/decoder/binary_dumper_version3.py -d -e

# Plot summary dashboards (plotly)
uv run python src/samidare_util/decoder/binary_dumper_version3.py -p

# Limit scanned bytes
uv run python src/samidare_util/decoder/binary_dumper_version3.py -d -l 10000000
```


## PySpark jobs

Some scripts use PySpark. Run them via **spark-submit** (through uv):

```bash
uv run spark-submit path/to/your_job.py
```

> Prefer `uv run spark-submit` (not `uv run pyspark`) so you can pass Spark configs like memory, cores, etc.

### Handling large data

- Avoid `toPandas()` / `collect()` on large DataFrames (risk of OOM).
- Inspect samples with `df.limit(5).show(truncate=False)`, `head(1)`, or `take(n)`.
- If you must iterate on the driver, use `toLocalIterator()` and keep it minimal.


## Others

### Pulse extraction (outline)

- Baseline correction by channel (subtract mode/median per channel) via UDF.
- Rising/falling thresholds with pre/post buffers to cut segments (pandas UDF).
- Multi-pulse per row → return `array<array<long>>` fields.

Keep the logic columnar (Spark UDF / pandas UDF) and avoid collecting raw arrays to the driver.


## Configuration

`parameters.toml` (example):

```toml
[format]
header_marker = 0xafaf
footer_marker = 0xfafa
t1_marker     = 0xfffa
t2_marker     = 0xfaaf
t3_marker     = 0xaffa

[fileinfo]
base_input_path  = "/path/to/data"
input_file_dir   = "raw"
input_file_name  = "SAM_MiniTPC_..." # without .bin
base_output_path = "/path/to/output"
```

## Troubleshooting

- **Conda active?** → `conda deactivate` before using uv.
- **Spark OOM / crashes**:
  - Don’t use `collect()` / `toPandas()` on large data.
  - Tune memory:  
    `uv run spark-submit --conf spark.executor.memory=4g --conf spark.driver.memory=4g your_job.py`
  - Validate the pipeline on a small sample first.


## Quick command cheatsheet

```bash
# Run a normal Python script
uv run python path/to/script.py [options]

# Run a PySpark job
uv run spark-submit path/to/job.py [spark options] [-- job options]
```

# Binary → Parquet Pipeline (logger1 / logger2)

`binary_dumper_version3.py` turns a raw byte stream into **two Parquet datasets** using two writers:

- **logger1** → `SAMPADataParquetAppender` → `*_raw.parquet`  
  Row-per-**block** (lowest-level unit).
- **logger2** → `PulseParquetAppender` → `*_event.parquet`  
  Row-per-**contiguous time series** per chip.


## How the binary is parsed

The main loop `scan_stream(...)` walks the input stream byte-by-byte while trying **all 8 bit-shifts** (0..7) to handle possible bit misalignment. A block is defined as a region that starts at **header**  and (ideally) ends at **footer** within a **60‑byte window** right after the header. If a footer is found, the two bytes immediately after the footer must equal the first two bytes after the header; otherwise the block is treated as abnormal. If the footer is not found within the 60‑byte window, the code falls back to splitting at the next header.

### Inside each block

- **Timestamps** are reconstructed from three 2‑byte markers that may appear in the block:
  - `0xAFFA`, `0xFAAF`, `0xFFFA` → take the 2 bytes *after* each marker, then combine as `(t3 << 32) | (t2 << 16) | t1` to get a 48‑bit timestamp.
- The first two bytes **after the block header** encode:
  - **chip** (1 byte)
  - **sample_index** (1 byte)
- **10‑bit samples**: from the first occurrence of `0xFFFA` (+4 bytes) up to the footer `0xFAFA`, the payload is unpacked as a stream of **10‑bit integers** (0..1023), MSB‑first (see `expand_10bit_units_from_pairs`).

The error flag is calculated from five flags: `datasize_flag`, `timestamp1_flag`, `timestamp2_flag`, `timestamp3_flag`, and `footer_flag` as decimal value.
If True, they are represented as a binary number with 0 for True and 1 for False, packed into bit i, and then converted to a decimal value.

## Stage 1 — logger1 (`SAMPADataParquetAppender`) → `*_raw.parquet`

**Purpose:** Store one row per **detected block** (lowest-level), including metadata and the unpacked **1D** sample vector. This is the best for low-level data check.

**Schema**

| field         | type               | meaning                                                          |
|---------------|--------------------|------------------------------------------------------------------|
| `data_block`  | `long`             | Effective block size in bytes (`gap_size + 2`)                   |
| `error_level` | `long`             | `0` if all markers found; higher values mean missing markers     |
| `timestamp`   | `long`             | 48‑bit timestamp reconstructed from markers                      |
| `chip`        | `long`             | Chip id (0..31 etc.)                                             |
| `sample_index`| `long`             | Sample index (e.g., 0..63)                                       |
| `samples_value` | `array<long>`    | Unpacked 10‑bit **1D** sample                                    |

**Example (conceptual)**

```json
{
  "data_block": 60,
  "error_level": 0,
  "timestamp": 123456789012,
  "chip": 1,
  "sample_index": 7,
  "samples_value": [512, 520, 515, ...]
}
```

## Stage 2 — grouping contiguous rows → logger2 (`PulseParquetAppender`) → `*_event.parquet`
Reconstruct the pulse for each channel from stream data blocks in `sample_builder()` and store it as one block.
`sample_builder()` is received only **normal** (`error_level == 0` and **60‑byte** blocks ) from `emit_block()`.

Rows are grouped as long as they keep the **same `chip`** and **strictly increasing `sample_index`**. When this continuity breaks, the group is “closed”, reshaped into the list of samples, and written by **logger2**.

**Schema**
| field           | type                       | meaning                                    |
|-----------------|----------------------------|--------------------------------------------|
| `chip`          | `long`                     | Chip id of this contiguous group           |
| `timestamp`     | `array<long>`              | Lost of timestamps                         |
| `sample_index`  | `array<long>`              | row sample indices                         |
| `samples_value` | `array<array<long>>`       | list of pulses as `[channel][sample value]`|

**Example (conceptual)**

```json
{
  "chip": 3,
  "timestamp": [621, 653, 685, ...],
  "sample_index": [0, 1, 2, ... 63],
  "samples_value": [
    [501, 498, 502, ...],   // channel 0 pulse
    [620, 615, 618, ...],   // channel 1 pulse
    ...
[620, 615, 618, ...], // channel 31 pulse
  ]
}
```