"""!
@file pyspark_noise_check.py
@version 1
@author Fumitaka ENDO
@date 2025-09-16T22:40:52+09:00
@brief Check noise level
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
import samidare_lib.decoder.binary_dumper_version3 as bd
import samidare_lib.decoder.pyspark_pulse_analysis_version2 as pau
import samidare_lib.analysis.pyspark_hit_pattern as hit
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

    # vals = (
    # df.select(F.explode_outer("samples_value").alias("v"))   # 各要素を行に展開
    #   .where(F.col("v").isNotNull())               # null要素は除外（必要なら）
    #   .agg(F.collect_list("v").alias("values"))    # 1つの配列に再結合
    #   .first()["values"]                           # Pythonリストとして取得
    # )

    data0 = df.filter(F.col("chip")==0).toPandas()
    sample_vals = data0["samples_value"]  # 各行: 32ch x 64samples の二重リスト

    # # すべてのイベント・全ch・全サンプルを一次元にフラット化
    # valid_val = [v for ev in sample_vals for ch in ev for v in ch]  # ← これが一次元の数値リスト

    # # 必要なら型・範囲を整理
    # valid_val = [int(v) for v in valid_val if v is not None and 0 <= v <= 1023]

    # counts1, centers1, edges1 = hit.make_hist(valid_val, nbins=1024, data_range=(0, 1024))
    # params1, info1 = hit.fit_gaussian(centers1, counts1, fit_range=(0, 1024))

    # fitx1 = np.linspace(0, 100,400)
    # fity1 = hit._gauss(fitx1, params1["A"], params1["mu"], params1["sigma"])

    # fig = make_subplots(rows=1, cols=1, vertical_spacing=0.15, horizontal_spacing=0.1,
    #         subplot_titles=(f"noise @ ch 15, mu: {params1["mu"]:.2f}, sigma: {params1["sigma"]:.2f}, sigma/mu: {params1["sigma"]/params1["mu"]*100.:.2f}",))
    # pau.add_sub_plot(fig,1,1,'spark-hist',[centers1,counts1],['Pulse hieght [ch]','Counts'])
    # pau.add_sub_plot(fig,1,1,'fit',[fitx1, fity1])
    # fig.update_layout( height=800, width=1000, showlegend=False,title_text=f"{input_finename}")
    # fig.show()

    # 0) 前提
    rows, cols = 8, 4          # 36 面子分の枠
    n_channels = 32            # チャンネル数
    S = 8                      # 1イベントで拾うサンプル数（先頭 S 点）

    # 1) 2次元配列を空で初期化（各chごとに入れていく箱を作る）
    nbaselines = [[] for _ in range(n_channels)]
    nmaxsamples = [[] for _ in range(n_channels)]


    enc_x=[]
    enc_y=[]
    

    # 2) 先頭 S サンプルを各イベント・各chから集める
    #    sample_vals[ev][ich] が「そのイベントのそのchのサンプル配列」を想定
    for ev in sample_vals:
        if ev is None:
            continue
        for ich, ch in enumerate(ev):
            if ich >= n_channels or ch is None:
                continue
            # ch[:S] は長さが S 未満でも安全に動く
            ble = ch[:S]
            nbaselines[ich].extend(ble)   
            nmaxsamples[ich].append(max(ch)-sum(ble)/len(ble))


    max_vsp = 0.98 / (rows - 1)        # ≒ 0.14
    max_hsp = 0.98 / (cols - 1)        # ≒ 0.49

    vsp = min(0.10, max_vsp)           # 0.10 など十分小さく
    hsp = min(0.06, max_hsp)

    fig = make_subplots(
        rows=rows, cols=cols,
        vertical_spacing=vsp,
        horizontal_spacing=hsp
    )

    for idx in range(min(n_channels, rows * cols)):
        r = idx // cols + 1   # 6 列で割る
        c = idx %  cols + 1

        pau.add_sub_plot(
            fig, r, c, '1d',
            [nbaselines[idx]],
            [f'Channel {idx} values', 'Counts'],
            xrange=[0,1024,1],                                # （あなたの関数の引数に合わせて）
            dataname=f"baseline chip, ch: {(0, idx)}"
        )

        counts1, centers1, edges1 = hit.make_hist(nbaselines[idx], nbins=1024, data_range=(0, 1024))
        params1, info1 = hit.fit_gaussian(centers1, counts1, fit_range=(0, 1024))
        fitx1 = np.linspace(0, 200,800)
        fity1 = hit._gauss(fitx1, params1["A"], params1["mu"], params1["sigma"])
        pau.add_sub_plot(fig, r, c,'fit',[fitx1, fity1])

        # print(f"chip, ch: {(0, idx)}, mean: {params1["mu"]}, sigma: {params1["sigma"]}, ratio: {params1["sigma"]/params1["mu"]}")

        ch2mv = 2000./1024.
        gQ2V  = 20.

        print(f"chip, ch: {(0, idx)}, sigma: {params1["sigma"]}, ENC: {6241*ch2mv*params1["sigma"]/gQ2V}") 

        enc_x.append(idx)
        enc_y.append(6241*ch2mv*params1["sigma"]/gQ2V)


        # pau.add_sub_plot(
        #     fig, r, c, '1d',
        #     [nmaxsamples[idx]],
        #     [f'Channel {idx} values', 'Counts'],
        #     xrange=[0,1024,1],                                 # （あなたの関数の引数に合わせて）
        #     dataname=f"maxsample chip, ch: {(0, idx)}"
        # )

        # counts1, centers1, edges1 = hit.make_hist(nmaxsamples[idx], nbins=1024, data_range=(0, 1024))
        # params1, info1 = hit.fit_gaussian(centers1, counts1, fit_range=(0, 1024))
        # fitx1 = np.linspace(0, 1024,6000)
        # fity1 = hit._gauss(fitx1, params1["A"], params1["mu"], params1["sigma"])
        # pau.add_sub_plot(fig, r, c,'fit',[fitx1, fity1])

    if 0:
        fig.update_layout(height=1000, width=800, showlegend=False, title_text=f"{input_finename}")
        fig.show()

    if 1:
        fig = make_subplots(rows=1, cols=1, vertical_spacing=0.15, horizontal_spacing=0.1,
                subplot_titles=(f"Pulse height @ chip0 ch31, mu: {params1["mu"]:.2f}, sigma: {params1["sigma"]:.2f}, sigma/mu: {params1["sigma"]/params1["mu"]*100.:.2f}",))
        pau.add_sub_plot(fig,1,1,'spark-hist',[centers1,counts1],['Channel [ch]','Counts'])
        pau.add_sub_plot(fig,1,1,'fit',[fitx1, fity1])
        fig.update_layout( height=600, width=1000, showlegend=False,title_text=f"{input_finename}")
        fig.show()

    if 1:
        fig = make_subplots(rows=1, cols=1, vertical_spacing=0.15, horizontal_spacing=0.1,
                subplot_titles=(f"ENC level @ chip 0",))
        pau.add_sub_plot(fig,1,1,'plot',[enc_x,enc_y],['Chip Channel','ENC [e-]'], color='red')
        fig.update_layout( height=600, width=1000, showlegend=False,title_text=f"{input_finename}")
        fig.show()

if __name__ == '__main__':
    main()

