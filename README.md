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
uv run python src/samidare_util/decoder/binary_dumper_version3.py --d -l 10000000
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



