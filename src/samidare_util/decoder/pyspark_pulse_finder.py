"""!
@file pyspark_pulse_finder.py
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
import os
import math
import matplotlib.pyplot as plt
from typing import Iterator, List
from pyspark.sql import SparkSession, Row, functions as F, types as T
from pyspark.sql.functions import pandas_udf
from pyspark.sql import functions as F, Window
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from itertools import islice
import samidare_util.decoder.binary_dumper_version3 as bd

this_file_path = pathlib.Path(__file__).parent
sys.path.append(str(this_file_path.parent.parent.parent / "src"))

def common_options(func):
    @click.option('--rise'    , '-r'  , type=int, default=50, help='rising thresold')
    @click.option('--fall'    , '-f'  , type=int, default=10, help='falling thresold')
    @click.option('--pren'    , '-b'  , type=int, default=10, help='number of pre sample')
    @click.option('--postn'   , '-a'  , type=int, default=8, help='number of post sample')
    @click.option('--minlen'  , '-l'  , type=int, default=4, help='minimum pulse length')
    @click.option('--maxevt'  , '-m'  , type=int, default=5000, help='maximum event number for debug')
    @click.option('--checkts' , '-cts', is_flag=True, help='check timestamp data flag')
    @click.option('--checkpf' , '-cpf', is_flag=True, help='check pulse finder result flag')
    @click.option('--duration', '-dt' , type=float, default=1, help='valid timing duration for plot pulse')
    @click.option('--save'            , is_flag=True, help='output file generation flag')
    @click.option('--refch'   , '-c'  , type=int, default=0, help='reference channel for timastamp plot')
    @click.option('--file'   , type=str, default=None, help='file name without .bin')
    @click.option('--dir'    , type=str, default=None, help='base directory name')
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@F.pandas_udf(T.ArrayType(T.ArrayType(T.LongType())))
def baseline_subtraction_first10(sv: pd.Series) -> pd.Series:
    """
    sv: 各行が 2次元配列 (list[list[int]] or ndarray[object])
    先頭10サンプルの平均をベースラインとして減算（四捨五入→int64）して返す
    """
    out_rows = []

    for row in sv:
        # 行が None/空
        if row is None:
            out_rows.append([])
            continue

        # row を反復しやすいリストへ正規化
        row_iter = row.tolist() if isinstance(row, np.ndarray) else row
        try:
            n_items = len(row_iter)
        except TypeError:
            out_rows.append([])
            continue
        if n_items == 0:
            out_rows.append([])
            continue

        row_out = []
        for wave in row_iter:
            if wave is None:
                row_out.append([])
                continue

            # wave も正規化して数値配列へ
            w = wave.tolist() if isinstance(wave, np.ndarray) else wave
            try:
                x = np.asarray(w, dtype=np.int64)
            except Exception:
                # 非数が混じる等の保険
                x = pd.Series(w, dtype="float64").fillna(0).astype(np.int64).values

            if x.size == 0:
                row_out.append([])
                continue

            k = min(10, x.size)
            baseline = float(np.mean(x[:k]))  # 先頭kの平均
            y = np.rint(x - baseline).astype(np.int64).tolist()
            row_out.append(y)

        out_rows.append(row_out)

    return pd.Series(out_rows)

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

            try:
                if x.size and x.min() >= 0 and (x.max() - x.min()) <= 4096:
                    shifted = x - x.min()
                    baseline = int(np.bincount(shifted).argmax() + x.min())
                else:
                    baseline = int(pd.Series(x).mode().iloc[0])
            except Exception:

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


pulse_item = T.StructType([
    T.StructField("pulse",     T.ArrayType(T.LongType()), True),
    T.StructField("index",     T.ArrayType(T.LongType()), True),
    T.StructField("time", T.ArrayType(T.LongType()), True),  
    T.StructField("chip",      T.LongType(), True),
    T.StructField("channel",   T.LongType(), True),
])
pulse_array_schema = T.ArrayType(pulse_item)

def make_find_pulse_udf_v2(RISE_THR=300, FALL_THR=200, PRE_BUF=4, POST_BUF=8, MIN_LEN=3):
    @pandas_udf(pulse_array_schema)
    def find_pulse_v2(sv_col: pd.Series, ts_col: pd.Series, chip_col: pd.Series) -> pd.Series:
        outs = []
        for wave2d, ts, chip in zip(sv_col, ts_col, chip_col):
            if wave2d is None or len(wave2d) == 0 or ts is None:
                outs.append([]); continue
            ts = np.asarray(ts, dtype=np.int64)
            rows = []
            for ch_id, wave in enumerate(wave2d):
                if wave is None:
                    continue
                x = np.asarray(wave, dtype=np.int64)
                n = min(len(x), len(ts))
                x = x[:n]; tsn = ts[:n]
                i=0; in_pulse=False; start_idx=0
                while i < n:
                    if not in_pulse:
                        if x[i] >= RISE_THR:
                            in_pulse=True; start_idx=max(0, i-PRE_BUF)
                        i += 1
                    else:
                        if x[i] <= FALL_THR:
                            end_idx=min(n, i+1+POST_BUF)
                            if end_idx-start_idx >= MIN_LEN:
                                rows.append({
                                  "pulse": x[start_idx:end_idx].tolist(),
                                  "index": list(range(start_idx, end_idx)),
                                  "time": tsn[start_idx:end_idx].tolist(),
                                  "chip": int(chip),
                                  "channel": int(ch_id),
                                })
                            in_pulse=False; i=end_idx
                        else:
                            i += 1
                if in_pulse:
                    end_idx=n
                    if end_idx-start_idx >= MIN_LEN:
                        rows.append({
                          "pulse": x[start_idx:end_idx].tolist(),
                          "index": list(range(start_idx, end_idx)),
                          "time": tsn[start_idx:end_idx].tolist(),
                          "chip": int(chip),
                          "channel": int(ch_id),
                        })
            outs.append(rows)
        return pd.Series(outs)
    return find_pulse_v2



pulse_item3 = T.StructType([
    T.StructField("original",  T.ArrayType(T.LongType()), True),
    T.StructField("pulse",     T.ArrayType(T.LongType()), True),
    T.StructField("index",     T.ArrayType(T.LongType()), True),
    T.StructField("time",      T.ArrayType(T.LongType()), True),  
    T.StructField("chip",      T.LongType(), True),
    T.StructField("channel",   T.LongType(), True),
])
pulse_array_schema3 = T.ArrayType(pulse_item3)

def make_find_pulse_udf_v3(RISE_THR=300, FALL_THR=200, PRE_BUF=4, POST_BUF=8, MIN_LEN=3):
    @pandas_udf(pulse_array_schema3)
    def find_pulse_v3(sv_col: pd.Series, ts_col: pd.Series, chip_col: pd.Series) -> pd.Series:
        outs = []
        for wave2d, ts, chip in zip(sv_col, ts_col, chip_col):
            if wave2d is None or len(wave2d) == 0 or ts is None:
                outs.append([]); continue
            ts = np.asarray(ts, dtype=np.int64)
            rows = []
            for ch_id, wave in enumerate(wave2d):
                if wave is None:
                    continue
                x = np.asarray(wave, dtype=np.int64)
                n = min(len(x), len(ts))
                x = x[:n]; tsn = ts[:n]
                i=0; in_pulse=False; start_idx=0
                while i < n:
                    if not in_pulse:
                        if x[i] >= RISE_THR:
                            in_pulse=True; start_idx=max(0, i-PRE_BUF)
                        i += 1
                    else:
                        if x[i] <= FALL_THR:
                            end_idx=min(n, i+1+POST_BUF)
                            if end_idx-start_idx >= MIN_LEN:
                                rows.append({
                                  "original": x.tolist(),
                                  "pulse": x[start_idx:end_idx].tolist(),
                                  "index": list(range(start_idx, end_idx)),
                                  "time": tsn[start_idx:end_idx].tolist(),
                                  "chip": int(chip),
                                  "channel": int(ch_id),
                                })
                            in_pulse=False; i=end_idx
                        else:
                            i += 1
                if in_pulse:
                    end_idx=n
                    if end_idx-start_idx >= MIN_LEN:
                        rows.append({
                          "original": x.tolist(),
                          "pulse": x[start_idx:end_idx].tolist(),
                          "index": list(range(start_idx, end_idx)),
                          "time": tsn[start_idx:end_idx].tolist(),
                          "chip": int(chip),
                          "channel": int(ch_id),
                        })
            outs.append(rows)
        return pd.Series(outs)
    return find_pulse_v3

def dumps_pretty_inline_arrays(obj, inline=('timestamp','sample_index')):
    """obj を indent=2 で整形しつつ、inline に挙げた配列だけ 1 行にする。"""
    s = json.dumps(obj, ensure_ascii=False, indent=2, separators=(',', ': '))
    for k in inline:
        v = obj.get(k)
        if isinstance(v, list):
            arr = json.dumps(v, ensure_ascii=False, separators=(', ', ': '))
            s = re.sub(rf'"{k}":\s*\[[^\]]*\]', f'"{k}": {arr}', s, flags=re.S)
    return s

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@common_options
def main(rise, fall, pren, postn, minlen, maxevt, checkts, checkpf, duration, save, refch, file, dir):

    toml_file_path = this_file_path  / "../../../parameters.toml"

    with open(toml_file_path, "r") as f:
        config = toml.load(f)

    fileinfo = config["fileinfo"]

    if dir is not None:
        fileinfo["base_output_path"] = dir

    if file is not None:
        fileinfo["input_file_name"] = file

    base_path = fileinfo["base_output_path"]  + "/" + fileinfo["input_file_name"] 
    input1 = base_path + "_raw.parquet"
    input2 = base_path + "_event.parquet"
    output = base_path + "_pulse.parquet"

    input_finename = os.path.basename(input2)

    spark = (
        SparkSession.builder
        .config("spark.driver.memory", "8g")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "128") 
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .config("spark.sql.files.maxPartitionBytes", 32 * 1024 * 1024)
        .getOrCreate()
    )

    df = (spark.read.parquet(input2).select("chip", "timestamp", "samples_value"))

    df2 = (df
           .withColumn("pulse_sub", baseline_subtraction_first10(F.col("samples_value")))
           .withColumn("row_id", F.monotonically_increasing_id())
           .filter( (F.col("row_id") > 1) ) )


    find_pulse_v3 = make_find_pulse_udf_v3(RISE_THR=rise, FALL_THR=fall, PRE_BUF=pren, POST_BUF=postn, MIN_LEN=minlen)

    df_pulses = (
        df2 
        .select( 
            find_pulse_v3(F.col("pulse_sub"), F.col("timestamp"), F.col("chip")).alias("items")
        )
        .select(F.explode_outer("items").alias("p"))      
        .select(
            F.col("p.original").alias("original_pulse"),
            F.col("p.pulse").alias("pulse"),
            F.col("p.index").alias("pulse_index"),
            F.col("p.time").alias("pulse_timestamp"),    
            F.col("p.chip").alias("chip"),
            F.col("p.channel").alias("channel"),
        )
    )

    df_plot = (
        df_pulses
        .withColumn("t0_ms", F.element_at("pulse_timestamp", 1).cast("double")/F.lit(320000.0))
        # .withColumn("t0_ms", F.element_at("pulse_timestamp", 1)/F.lit(32.00) * F.lit(3.125) / F.lit(1e6))
        .where(F.col("t0_ms").isNotNull())
        .select("chip", "channel", "t0_ms", "pulse", "pulse_index", "pulse_timestamp","original_pulse")  # ← ここで保持
        .orderBy("t0_ms")
    )

    if 0:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
        for row in islice(df_plot.toLocalIterator(), maxevt):  # ← 5000件だけ
            x       = row["pulse_index"]
            y       = row["pulse"]
            oy      = row["original_pulse"]
            id = row["channel"]
            ox=[]

            for i in range(len(oy)):
                ox.append(i)

            if id ==15:
                ax.plot(ox,oy,lw=1, alpha=0.5, marker="o", markersize=2, label=f"original", color='black')
                ax.plot(x,y,lw=1, alpha=0.5, marker="o", markersize=2, label=f"found pulse", color='red')

                ax.set(xlim=(0, 64), ylim=(-50, 950))
                ax.set_title(f"pulse @ chip{0}")     
                ax.set_xlabel("Sample index")
                ax.set_ylabel("Sample value - Base line(mode value)")   
                ax.legend(loc='upper right', ncol=3, fontsize=10)
                plt.show(block=True)  



    if checkts:
        chip_list, ts_list = [], []

        for row in islice(df_plot.toLocalIterator(), maxevt):  # ← 5000件だけ
            if row["channel"] == refch:
                chip_list.append(int(row["chip"]))
                ts_list.append(float(row["t0_ms"]))

        fig = go.Figure(go.Scattergl(x=ts_list, y=chip_list, mode="markers", name="test"))
        fig.update_layout(title=f"input file: {input_finename}.bin, pulse counts: {maxevt}, channel: {refch}", xaxis_title="timestamp/320000 ~ [ms] ?", yaxis_title="chip", height=800)
        fig.show() 

        fig = go.Figure(go.Scattergl(y=ts_list, mode="markers+lines", name="test"))
        fig.update_layout(title=f"input file: {input_finename}.bin, pulse counts: {maxevt}, channel: {refch}", yaxis_title="timestamp/320000 ~ [ms] ?", xaxis_title="index", height=800)
        fig.show() 

    if checkpf:

        fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
        ax = axes.ravel()
        for i in range(4):
            ax[i].set_title(f"pulse @ chip{i}")
            ax[i].set(xlim=(0, 64), ylim=(-300, 725))
            ax[i].set_xlabel("Sample index")
            ax[i].set_ylabel("Sample value - Base line(mode value)")

        pt = -1

        for row in islice(df_plot.toLocalIterator(), maxevt):  # ← 5000件だけ
            chip    = int(row["chip"])
            channel = int(row["channel"])
            ct      = float(row["t0_ms"])  
            x       = row["pulse_index"]
            y       = row["pulse"]

            if ct is None:
                continue

            if (pt == -1) or (abs(ct - pt) < duration):   
                ax[chip].plot(x, y, lw=1, alpha=0.5, marker="o", markersize=2, label=f"ch{channel}", color=bd.color32(channel, "brg"))
            else:
                ax[chip].plot(x, y, lw=1, alpha=0.5, marker="o", markersize=2, label=f"ch{channel}", color=bd.color32(channel, "brg"))
                
                for i in range(4):
                    ax[i].legend(loc='upper right', ncol=3, fontsize=6)
                
                plt.show(block=False)
                plt.pause(0.1)
                
                for i in range(4):
                    ax[i].clear()
                    ax[i].set(xlim=(0, 64), ylim=(-300, 725))
                    ax[i].set_title(f"pulse @ chip{i}")
                    ax[i].set_xlabel("Sample index")
                    ax[i].set_ylabel("Sample value - Base line(mode value)")

            pt = ct if ct != pt else pt  

    if save:
        (df_plot
            .select("chip", "channel", "t0_ms", "pulse", "pulse_index", "pulse_timestamp")
            .write
            .mode("overwrite")
            .option("compression", "zstd")
            .parquet(output))
    

    df_plot.select("chip", "channel", "pulse", "pulse_index", "pulse_timestamp", "t0_ms").show(20)


if __name__ == '__main__':
    main()
