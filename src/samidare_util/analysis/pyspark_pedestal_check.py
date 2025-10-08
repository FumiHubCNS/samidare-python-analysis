"""!
@file pyspark_pedestal_check.py
@version 1
@author Fumitaka ENDO
@date 2025-10-07T18:58:01+09:00
@brief check noise level
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
import samidare_util.decoder.pyspark_pulse_analysis_version2 as pau
import samidare_util.analysis.pyspark_hit_pattern as hit
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
from pyspark.sql import functions as F, types as T, Window

this_file_path = pathlib.Path(__file__).parent
sys.path.append(str(this_file_path.parent.parent.parent / "src"))

def common_options(func):
    @click.option('--name', '-n', default='World', show_default=True, help='Your name')
    @click.option('--date', '-d', type=click.DateTime(formats=["%Y-%m-%d"]), default=lambda: datetime.datetime.now(), show_default=lambda: datetime.datetime.now().strftime("%Y-%m-%d"), help="Date in YYYY-MM-DD format")
    @click.option('--verbose', '-v', is_flag=True, help='verbose flag')
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def rms_np(values):
    a = np.asarray(values, dtype=float)
    a = a[~np.isnan(a)]            # NaN除去
    if a.size == 0:
        raise ValueError("空のデータです")
    return np.sqrt(np.mean(a*a))


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

    # パラメータ
    S = 64                  # pedestal 計算に使う先頭サンプル数
    NBINS = 200            # ヒストグラムのビン数
    LOW, HIGH = 0.0, 1024.0  # RMS の集計範囲（必要に応じて調整）

    # 必要列だけ読む（例: chip, samples_value）
    df0 = df.select("chip", "samples_value")

    # 1) 各行の各チャンネルの先頭 S サンプルを切り出し
    df1 = df0.withColumn(
        "heads",
        F.expr(f"transform(samples_value, c -> slice(c, 1, {S}))")   # array<array<...>>
    )

    # 2) heads から pedestal 平均（μ）をチャンネル毎に配列で計算
    df2 = df1.withColumn(
        "ped_mean",
        F.expr("""
        transform(
            heads,
            h -> aggregate(transform(h, x -> double(x)), 0D, (acc, x) -> acc + x) / greatest(size(h), 1)
        )
        """)
    )

    # 3) pedestal RMS（σ）をチャンネル毎に配列で計算
    #    arrays_zip で (heads, ped_mean) を束ね、各要素（=チャンネル）で sqrt( Σ(x-μ)^2 / N )
    df3 = df2.withColumn(
        "ped_rms",
        F.expr("""
        transform(
            arrays_zip(heads, ped_mean),
            z -> sqrt(
                aggregate(
                    transform(z.heads, x -> double(x)),
                    0D,
                    (acc, x) -> acc + pow(x - z.ped_mean, 2)
                ) / greatest(size(z.heads), 1)
                )
        )
        """)
    ).drop("heads")  # heads はもう不要

    # 4) 配列を (chip, channel, rms) の縦持ちへ
    rms_long = (
        df3
        .select("chip", F.posexplode("ped_rms").alias("channel", "rms"))  # channel=0..31
        .where(F.col("rms").isNotNull())
    )

    ch2mv = 2000.0/1024.0
    gQ2V  = 20.

    enc_df = (
        rms_long
        .withColumn("enc", F.lit(6241.0) * F.lit(ch2mv) * F.col("rms") / F.lit(gQ2V))
        .withColumn("gch", F.col("chip") * 32 + F.col("channel") )
    )

    if 0:

        chip_id, ch_id = 0, 0
        df_filter = (
            enc_df
            .where((F.col("chip")==chip_id) & (F.col("channel")==ch_id))
        )

        data = df_filter.toPandas()
        counts1, centers1, edges1 = hit.make_hist(data["rms"], nbins=50, data_range=(0, 5))
        rms_val = rms_np(data["rms"])
        params1, info1 = hit.fit_gaussian(centers1, counts1, fit_range=(0, 5))
        fitx1 = np.linspace(0, 5,800)
        fity1 = hit._gauss(fitx1, params1["A"], params1["mu"], params1["sigma"])

        fig = make_subplots(rows=1, cols=1, vertical_spacing=0.15, horizontal_spacing=0.1,
                subplot_titles=(f"Pulse height @ chip {chip_id} ch {ch_id}, mu: {params1["mu"]:.2f}, sigma: {params1["sigma"]:.2f}, rms: {rms_val:.2f}",))
        pau.add_sub_plot(fig,1,1,'spark-hist',[centers1,counts1],['Channel [ch]','Counts'])
        pau.add_sub_plot(fig,1,1,'fit',[fitx1, fity1])
        fig.update_layout( height=600, width=1000, showlegend=False,title_text=f"{input_finename}")
        fig.show()


    if 0:
        data = enc_df.toPandas()
        fig = make_subplots(rows=1, cols=1, vertical_spacing=0.15, horizontal_spacing=0.1,
                subplot_titles=(f"pedestal",))
        pau.add_sub_plot(fig,1,1,'2d',[data["gch"],data["rms"]],['Channel','ADU rms'])
        fig.update_layout( height=600, width=1000, showlegend=False,title_text=f"{input_finename}")
        fig.show()

    if 1:
        plot_x = []
        plot_y = []
        enc_y = []

        for i in range(4):
            for j in range(32):
                gch = i * 32 + j
                chip_id, ch_id = i, j
                df_filter = (enc_df.where((F.col("chip")==chip_id) & (F.col("channel")==ch_id)))
                data = df_filter.toPandas()
                rms_val = rms_np(data["rms"]) 
                enc_val = rms_np(data["enc"]) 

                plot_x.append(gch)
                plot_y.append(rms_val)
                enc_y.append(enc_val)
        
        fig = make_subplots(rows=1, cols=2, vertical_spacing=0.15, horizontal_spacing=0.1,
            subplot_titles=(
                f"ADU rms, min: {min(plot_y):.2f}, max: {max(plot_y):.2f}, mean: {sum(plot_y)/len(plot_y):.2f}",
                f"ENC rms, min: {min(enc_y):.2f}, max: {max(enc_y):.2f}, mean: {sum(enc_y)/len(enc_y):.2f}"
                )
        )
        pau.add_sub_plot(fig,1,1,'plot',[plot_x,plot_y],['Chip Channel','ADU rms'], color='red')
        pau.add_sub_plot(fig,1,2,'plot',[plot_x,enc_y],['Chip Channel','ENC rms'], color='blue')
        fig.update_layout( height=600, width=1800, showlegend=False,title_text=f"{input_finename}")
        fig.show()


if __name__ == '__main__':
    main()

