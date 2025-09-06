"""!
@file pyspark_panquet_test.py
@version 1
@author FumiHubCNS
@date 2025-09-06T14:51:09+09:00
@brief template text
"""
import click
import pathlib
import datetime
import sys
import json
import toml
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from pyspark.sql import functions as F, types as T
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.functions import pandas_udf

this_file_path = pathlib.Path(__file__).parent
sys.path.append(str(this_file_path.parent.parent.parent / "src"))

def common_options(func):
    @click.option('--name', '-n', default='World', show_default=True, help='Your name')
    @click.option('--date', '-d', type=click.DateTime(formats=["%Y-%m-%d"]), default=lambda: datetime.datetime.now(), show_default=lambda: datetime.datetime.now().strftime("%Y-%m-%d"), help="Date in YYYY-MM-DD format")
    @click.option('--verbose', '-v', is_flag=True, help='verbose flag')
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@F.pandas_udf(T.ArrayType(T.ArrayType(T.LongType())))
def baseline_subtraction(sv: pd.Series) -> pd.Series:
    """!
    sv: 各行が 2次元配列 (list[list[int]]) を持つ Series
        例: [[ch0_samples...], [ch1_samples...], ...]
    戻り値: ベースライン(最頻値)減算後の 2次元配列
    """
    out = []
    for row in sv:
        if row is None or len(row) == 0:
            out.append([])
            continue

        row_out = []
        for wave in row:
            if wave is None or len(wave) == 0:
                row_out.append([])
                continue

            x = np.asarray(wave, dtype=np.int64)

            # ベースライン推定（最頻値優先、フォールバックあり）
            try:
                if x.size and x.min() >= 0 and (x.max() - x.min()) <= 4096:
                    # 値域が狭い非負整数なら bincount で高速に最頻値
                    shifted = x - x.min()
                    baseline = int(np.bincount(shifted).argmax() + x.min())
                else:
                    # 一般形（多峰性対応）: 最頻値の先頭を採用
                    baseline = int(pd.Series(x).mode().iloc[0])
            except Exception:
                # 最後の手段として中央値
                baseline = int(np.median(x))

            y = (x - baseline).astype(np.int64).tolist()
            row_out.append(y)

        out.append(row_out)

    return pd.Series(out)

pulse_out_schema = T.StructType([
    T.StructField("pulses",         T.ArrayType(T.ArrayType(T.LongType())), True),
    T.StructField("original_index", T.ArrayType(T.ArrayType(T.LongType())), True),
    T.StructField("timestamps",     T.ArrayType(T.ArrayType(T.LongType())), True),
    T.StructField("chip",           T.ArrayType(T.LongType()), True),
    T.StructField("channel",        T.ArrayType(T.LongType()), True),
])

def make_find_pulse_udf(RISE_THR=300, FALL_THR=200, PRE_BUF=4, POST_BUF=8, MIN_LEN=3):
    @pandas_udf(pulse_out_schema)
    def find_pulse(sv_col: pd.Series, ts_col: pd.Series, chip_col: pd.Series) -> pd.DataFrame:
        out_pulses, out_idx, out_ts, out_chip, out_ch = [], [], [], [], []
        for wave2d, ts, chip in zip(sv_col, ts_col, chip_col):
            if wave2d is None or len(wave2d) == 0 or ts is None:
                out_pulses.append([]); out_idx.append([]); out_ts.append([])
                out_chip.append([]);   out_ch.append([])
                continue

            ts = np.asarray(ts, dtype=np.int64)
            pulses_row, idx_row, ts_row, chip_row, ch_row = [], [], [], [], []

            for ch_id, wave in enumerate(wave2d):
                if wave is None or len(wave) == 0:
                    continue
                x = np.asarray(wave, dtype=np.int64)
                n = min(len(x), len(ts))
                x = x[:n]; tsn = ts[:n]

                i = 0; in_pulse = False; start_idx = 0
                while i < n:
                    if not in_pulse:
                        if x[i] >= RISE_THR:
                            in_pulse = True
                            start_idx = max(0, i - PRE_BUF)
                        i += 1
                    else:
                        if x[i] <= FALL_THR:
                            end_idx = min(n, i + 1 + POST_BUF)
                            if end_idx - start_idx >= MIN_LEN:
                                pulses_row.append(x[start_idx:end_idx].tolist())
                                idx_row.append(list(range(start_idx, end_idx)))
                                ts_row.append(tsn[start_idx:end_idx].tolist())
                                chip_row.append(int(chip))
                                ch_row.append(int(ch_id))
                            in_pulse = False
                            i = end_idx
                        else:
                            i += 1
                if in_pulse:
                    end_idx = n
                    if end_idx - start_idx >= MIN_LEN:
                        pulses_row.append(x[start_idx:end_idx].tolist())
                        idx_row.append(list(range(start_idx, end_idx)))
                        ts_row.append(tsn[start_idx:end_idx].tolist())
                        chip_row.append(int(chip))
                        ch_row.append(int(ch_id))

            out_pulses.append(pulses_row)
            out_idx.append(idx_row)
            out_ts.append(ts_row)
            out_chip.append(chip_row)
            out_ch.append(ch_row)

        return pd.DataFrame({
            "pulses": out_pulses,
            "original_index": out_idx,
            "timestamps": out_ts,
            "chip": out_chip,
            "channel": out_ch,
        })
    return find_pulse

def dumps_pretty_inline_arrays(obj, inline=('timestamp','sample_index')):
    """obj を indent=2 で整形しつつ、inline に挙げた配列だけ 1 行にする。"""
    s = json.dumps(obj, ensure_ascii=False, indent=2, separators=(',', ': '))
    for k in inline:
        v = obj.get(k)
        if isinstance(v, list):
            # 配列は「, 」で区切った 1 行の表現にする
            arr = json.dumps(v, ensure_ascii=False, separators=(', ', ': '))
            s = re.sub(rf'"{k}":\s*\[[^\]]*\]', f'"{k}": {arr}', s, flags=re.S)
    return s

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@common_options
def main(name, date, verbose):
    if verbose:
        click.echo(f"[VERBOSE MODE] Hello {name}, date: {date.strftime('%Y-%m-%d')}")
    else:
        click.echo(f"Hello {name}! (Date: {date.strftime('%Y-%m-%d')})")


    toml_file_path = this_file_path  / "../../../parameters.toml"

    with open(toml_file_path, "r") as f:
        config = toml.load(f)

    fileinfo = config["fileinfo"]
    base_path = fileinfo["base_output_path"]  + "/" + fileinfo["input_file_name"] 
    input1 = base_path + "_raw.parquet"
    input2 = base_path + "_event.parquet"

    spark = (SparkSession.builder
            .config("spark.driver.memory", "8g")
            .config("spark.sql.files.maxPartitionBytes", 64 * 1024 * 1024)  # 64MB
            .getOrCreate())

    df = (spark.read.parquet(input1))
    df.printSchema()
    df.limit(5).show(truncate=False)

    df = (spark.read.parquet(input2))
    df.printSchema()
    df.limit(5).show(truncate=False)


    df = (spark.read.parquet(input2).select("chip", "timestamp", "samples_value"))
    df2 = df.withColumn("pulse_sub", baseline_subtraction(F.col("samples_value")))

    find_pulse_udf = make_find_pulse_udf(RISE_THR=50, FALL_THR=50, PRE_BUF=10, POST_BUF=10, MIN_LEN=3)
    df3 = df2.withColumn("pulse", find_pulse_udf(F.col("pulse_sub"), F.col("timestamp"), F.col("chip")))
    df3 = (df3
        .withColumn("pulses",         F.col("pulse.pulses"))
        .withColumn("original_index", F.col("pulse.original_index"))
        .withColumn("timestamps",     F.col("pulse.timestamps"))
        .withColumn("pulse_chip",     F.col("pulse.chip"))
        .withColumn("pulse_channel",  F.col("pulse.channel"))
        .drop("pulse"))
    df3.printSchema()
    df3.limit(5).show(truncate=False)

    # パルスが1つ以上ある行を1行だけ取得
    row = (df3
        .filter(F.size("pulses") > 0)
        .select("chip", "pulses", "timestamps", "original_index",
                "pulse_chip", "pulse_channel")
        .head(4))
    if not row:
        raise RuntimeError("pulses を含む行が見つかりません。")
    row = row[3]

    plt.figure(figsize=(10,6))
    for i, (y, ts, idx, chip_p, ch_p) in enumerate(zip(
            row["pulses"] or [], 
            row["timestamps"] or [], 
            row["original_index"] or [],
            row["pulse_chip"] or [], 
            row["pulse_channel"] or [])):

        if not y: 
            continue

        if idx:
            x = idx
            xlabel = "original pulse list index"
        else:
            x = list(range(len(y)))
            xlabel = "found pulse index"

        n = min(len(x), len(y))
        plt.plot(x[:n], y[:n],  lw=1, alpha=0.5, marker="o", markersize=2, label=f"ch{ch_p}")

    plt.title(f"All pulses in a row (chip={row['chip']})")
    plt.xlabel(xlabel)
    plt.ylabel("amplitude - baseline")
    plt.legend(ncol=1, fontsize=10)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
