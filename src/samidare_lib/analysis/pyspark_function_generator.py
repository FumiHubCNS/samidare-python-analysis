"""!
@file pyspark_function_generator.py
@version 1
@author FumiHubCNS
@date 2025-09-11T20:36:45+09:00
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
from pyspark.sql import functions as F, Window as W
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from itertools import islice
import samidare_lib.decoder.binary_dumper_version3 as bd
import samidare_lib.decoder.pyspark_pulse_analysis_version2 as pau
import samidare_lib.analysis.pyspark_hit_pattern as hit
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
from pyspark.sql import functions as F, types as T, Window
import numpy as np
from typing import Iterable, Optional, List

this_file_path = pathlib.Path(__file__).parent
sys.path.append(str(this_file_path.parent.parent.parent / "src"))

def get_parquet_data(
    filename: str = None,
    N: int = 1000,
    clock: float = 320. 

):
    toml_file_path = this_file_path / "../../../parameters.toml"
    with open(toml_file_path, "r") as f:
        config = toml.load(f)

    fileinfo = config["fileinfo"]

    if filename is not None:
        base_path = fileinfo["base_output_path"] + "/" + filename
    else:
        base_path = fileinfo["base_output_path"] + "/" + fileinfo["input_file_name"] 

    input = base_path + "_event.parquet"
    input_finename = os.path.basename(input)
 
    spark = (
        SparkSession.builder
        .config("spark.driver.memory", "8g")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "128") 
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .config("spark.sql.files.maxPartitionBytes", 32 * 1024 * 1024)
        .getOrCreate()
    )

    df = spark.read.parquet(input)    

    # 1) 欲しい (chip, channel) の全ペアを作る（128行）
    pairs = [(c, ch) for c in range(4) for ch in range(32)]
    pairs_df = spark.createDataFrame(pairs, "chip long, channel long")

    # 2) 元データから必要な列だけ（例）
    base = df.select("chip","channel","t0_ms","pulse","pulse_timestamp")

    # 3) 実在するペアだけに絞り込む（pairsに含まれて、かつbaseに実在する行のみ）
    #    → pairs_df との inner join でOK（存在しないペアは落ちる）
    filtered = (
        base.join(F.broadcast(pairs_df), on=["chip","channel"], how="inner")
    )

    # 4) 各 (chip,channel) で N 件だけ取得（完全分散、順序は t0_ms 例）
    c_timestamp = clock * 1e3
    w = W.partitionBy("chip","channel").orderBy("t0_ms")
    df_limited = (
        filtered
        .withColumn("rn", F.row_number().over(w))
        .withColumn("maxsample", F.array_max(F.col("pulse")).cast("double")) 
        .withColumn("maxsample_index_raw", F.array_position(F.col("pulse"), F.array_max(F.col("pulse"))))
        .withColumn("maxsample_index", F.when(F.col("maxsample_index_raw") <= 0, F.lit(None).cast("int")).otherwise(F.col("maxsample_index_raw").cast("int")))
        .withColumn("maxsample_timing_ms", (F.element_at(F.col("pulse_timestamp"), F.col("maxsample_index")) / F.lit(c_timestamp)).cast("double"))
        .where(F.col("rn") <= N)
        .drop("rn")
    )

    # 事前にヌル除去（配列に None を混ぜない）
    df_clean = df_limited.where(
        F.col("maxsample_timing_ms").isNotNull() & F.col("maxsample").isNotNull()
    )

    # ヌル除去済み df_clean: (chip, channel, maxsample_timing_ms, maxsample, ...)
    agg = (
        df_clean
        .groupBy("chip", "channel")
        .agg(
            # time でソートするため、(t, m) の struct を collect してから sort_array
            F.sort_array(
                F.collect_list(
                    F.struct(
                        F.col("maxsample_timing_ms").alias("t"),
                        F.col("maxsample").alias("m")
                    )
                )
            ).alias("pairs"),
            F.count("*").alias("n")
        )
    )

    # pairs から timings / maxsamples を取り出す
    agg2 = (
        agg
        .withColumn("timings",    F.expr("transform(pairs, x -> x.t)"))
        .withColumn("maxsamples", F.expr("transform(pairs, x -> x.m)"))
        .drop("pairs")
    )

    # # （任意）rows 自体の並びもソートしたいなら：
    # #   1) chip,channel の順
    # rows = agg2.orderBy("chip", "channel").collect()

    # #   2) 先頭の timing でソートしたい場合
    # agg2b = agg2.withColumn("t0", F.element_at("timings", 1)).orderBy("t0")

    return agg2, input_finename

def trim_and_convert_ms_to_ns_np(
    values_ms: Iterable[float],
    lo_ms: Optional[float] = None,
    hi_ms: Optional[float] = None,
    inclusive: bool = True,
) -> List[float]:
    arr = np.asarray(values_ms, dtype=float)
    mask = np.isfinite(arr)
    if lo_ms is not None:
        mask &= (arr >= lo_ms) if inclusive else (arr > lo_ms)
    if hi_ms is not None:
        mask &= (arr <= hi_ms) if inclusive else (arr < hi_ms)
    return (arr[mask] * 1e6).tolist()


def main():
 
    toml_file_path = this_file_path  / "../../../parameters.toml"

    with open(toml_file_path, "r") as f:
        config = toml.load(f)

    fileinfo = config["fileinfo"]
    base_path = fileinfo["base_output_path"]  + "/" + fileinfo["input_file_name"] 
    input = base_path + "_pulse.parquet"
    input_finename = os.path.basename(input)
 
    spark = (
        SparkSession.builder
        .config("spark.driver.memory", "8g")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "128") 
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .config("spark.sql.files.maxPartitionBytes", 32 * 1024 * 1024)
        .getOrCreate()
    )

    df = spark.read.parquet(input)    

    # 1) 欲しい (chip, channel) の全ペアを作る（128行）
    pairs = [(c, ch) for c in range(4) for ch in range(32)]
    pairs_df = spark.createDataFrame(pairs, "chip long, channel long")

    # 2) 元データから必要な列だけ（例）
    base = df.select("chip","channel","t0_ms","pulse","pulse_timestamp")

    # 3) 実在するペアだけに絞り込む（pairsに含まれて、かつbaseに実在する行のみ）
    #    → pairs_df との inner join でOK（存在しないペアは落ちる）
    filtered = (
        base.join(F.broadcast(pairs_df), on=["chip","channel"], how="inner")
    )

    # 4) 各 (chip,channel) で N 件だけ取得（完全分散、順序は t0_ms 例）
    N = 1000
    w = W.partitionBy("chip","channel").orderBy("t0_ms")
    df_limited = (
        filtered
        .withColumn("rn", F.row_number().over(w))
        .withColumn("maxsample", F.array_max(F.col("pulse")).cast("double")) 
        .withColumn("maxsample_index_raw", F.array_position(F.col("pulse"), F.array_max(F.col("pulse"))))
        .withColumn("maxsample_index", F.when(F.col("maxsample_index_raw") <= 0, F.lit(None).cast("int")).otherwise(F.col("maxsample_index_raw").cast("int")))
        .withColumn("maxsample_timing_ms", (F.element_at(F.col("pulse_timestamp"), F.col("maxsample_index")) / F.lit(320000.0)).cast("double"))
        .where(F.col("rn") <= N)
        .drop("rn")
    )

    # 事前にヌル除去（配列に None を混ぜない）
    df_clean = df_limited.where(
        F.col("maxsample_timing_ms").isNotNull() & F.col("maxsample").isNotNull()
    )

    # ヌル除去済み df_clean: (chip, channel, maxsample_timing_ms, maxsample, ...)
    agg = (
        df_clean
        .groupBy("chip", "channel")
        .agg(
            # time でソートするため、(t, m) の struct を collect してから sort_array
            F.sort_array(
                F.collect_list(
                    F.struct(
                        F.col("maxsample_timing_ms").alias("t"),
                        F.col("maxsample").alias("m")
                    )
                )
            ).alias("pairs"),
            F.count("*").alias("n")
        )
    )

    # pairs から timings / maxsamples を取り出す
    agg2 = (
        agg
        .withColumn("timings",    F.expr("transform(pairs, x -> x.t)"))
        .withColumn("maxsamples", F.expr("transform(pairs, x -> x.m)"))
        .drop("pairs")
    )

    # （任意）rows 自体の並びもソートしたいなら：
    #   1) chip,channel の順
    rows = agg2.orderBy("chip", "channel").collect()

    #   2) 先頭の timing でソートしたい場合
    agg2b = agg2.withColumn("t0", F.element_at("timings", 1)).orderBy("t0")
    rows = agg2b.collect()

    hit1d = []

    N_CHIPS = 4
    N_CHS   = 32
    pulse_height_lists = [[[] for _ in range(N_CHS)] for _ in range(N_CHIPS)]


    if 1:
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.15, horizontal_spacing=0.1)

        for r in rows:
            chip = int(r["chip"]); ch = int(r["channel"])
            timings = r["timings"]        # List[float]
            maxamps = r["maxsamples"]     # List[float]
            if len(timings)>0:
                legend_option=[1.02,1,"left","top","v",160]
                

                pau.add_sub_plot(fig,1,1,'plot',
                        [timings, maxamps],
                        ['timestamp[ms]','max sample'],
                        legends=legend_option, 
                        dataname=f"chip, ch: {chip,ch}",
                        color=bd.color32(ch,as_hex=True, cmap='rainbow')
                    )
                
                pulse_height_lists[chip][ch].append(maxamps)

                pre  = np.array(timings[0:-1])
                post = np.array(timings[1:])
                timi = np.array(timings[1:])
                mdif = post - pre

                if chip == 0 and ch ==0:
                    hit1d = mdif

                
                pau.add_sub_plot(fig,2,1,'plot',
                        [timi,mdif],
                        ['timestamp[ms]','dt[ms]'], 
                        dataname=f"chip, ch: {chip,ch}",
                        color=bd.color32(ch,as_hex=True, cmap='rainbow')
                    )
        fig.update_xaxes(matches='x')
        fig.update_layout( height=800, width=1000, showlegend=False,title_text=f"{input_finename}")
        
        if 1:
            fig.show()


    if 0:
        figcounts=0
        fig = make_subplots(rows=6, cols=4, vertical_spacing=0.15, horizontal_spacing=0.1)

        for i in range(len(pulse_height_lists)):
            for j in range(len(pulse_height_lists[i])):
                if len(pulse_height_lists[i][j])>0:

                    # print(i,j,pulse_height_lists[i][j][0] )

                    pau.add_sub_plot(fig,(figcounts//4+1),(figcounts%4+1),'1d',
                                [pulse_height_lists[i][j][0]],
                                [f'{i},{j} [ch]','Counts'], [100],
                                dataname=f"chip, ch: {i,j}"
                            )
                
                    figcounts += 1

        fig.update_layout( height=800, width=1000, showlegend=False,title_text=f"{input_finename}")
        fig.show()
        

    if 0:
        # res = trim_and_convert_ms_to_ns_np(hit1d, lo_ms=0.99995, hi_ms=1.00005, inclusive=True)
        res = trim_and_convert_ms_to_ns_np(hit1d, lo_ms=0, hi_ms=2, inclusive=True)
        print(res)  # -> [950000.0, 1000000.0, 1200000.0]
        plt.hist(res, bins=100)
        plt.show()  

    pulse_sum = F.aggregate(F.col("pulse"), F.lit(0.0), lambda acc, x: acc + F.coalesce(x.cast("double"), F.lit(0.0)) ).alias("charge")

    if 0:
        df_test = (df_clean.withColumn("charge", pulse_sum).filter(F.col("channel")==15).filter(F.col("chip")==0))
        data = df_test.toPandas()

        counts1, centers1, edges1 = hit.make_hist(data["maxsample"], nbins=200, data_range=(0, 1024))
        params1, info1 = hit.fit_gaussian(centers1, counts1, fit_range=(0, 1024) )

        fitx1 = np.linspace(100, 300,400)
        fity1 = hit._gauss(fitx1, params1["A"], params1["mu"], params1["sigma"])

        fig = make_subplots(rows=1, cols=1, vertical_spacing=0.15, horizontal_spacing=0.1,
                subplot_titles=(f"Pulse height @ ch15, mu: {params1["mu"]:.2f}, sigma: {params1["sigma"]:.2f}, sigma/mu: {params1["sigma"]/params1["mu"]*100.:.2f}",))
        pau.add_sub_plot(fig,1,1,'spark-hist',[centers1,counts1],['Charge [ch]','Counts'])
        pau.add_sub_plot(fig,1,1,'fit',[fitx1, fity1])
        fig.update_layout( height=800, width=1000, showlegend=False,title_text=f"{input_finename}")
        fig.show()

if __name__ == '__main__':
    main()

