"""!
@file pyspark_pulse_check.py
@version 1
@author Fumitaka ENDO
@date 2025-09-07T16:17:11+09:00
@brief check data after pulse finding
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

def add_event_id_anchor(
    df,
    time_col: str = "t0_ms",
    threshold_ms: float = 50.0,
    id_col: str = "event_id",
):
    """
    タイムスタンプの昇順で走査し、「最後に閾値を超えた時刻（アンカー）」との差が
    threshold_ms を超えたらイベント番号を+1し、アンカーをその行に更新。
    keys なし（全体で単一の流れ）で番号を付与します。

    返り値: df に id_col を追加した DataFrame
    """

    # 1) 元DFに一意IDを付与（後で join で戻すため）
    df1 = df.withColumn("__rid", F.monotonically_increasing_id())

    # 2) t0_ms がある行だけを抽出して時刻昇順にソート
    df_nonnull = (
        df1
        .where(F.col(time_col).isNotNull())
        .select("__rid", F.col(time_col).cast("double").alias(time_col))
        .orderBy(F.col(time_col).asc(), F.col("__rid").asc())
        # 全体で連続走査したいので 1 partition にまとめる（データ量が非常に大きい場合は注意）
        .coalesce(1)
    )

    # 3) pandas 側でアンカー方式のイベントIDを付与（__rid と event_id だけ返す）
    schema_ids = T.StructType([
        T.StructField("__rid", T.LongType(), False),
        T.StructField(id_col, T.LongType(), False),
    ])

    def _assign_event_id(pdf_iter):
        for pdf in pdf_iter:
            if pdf.empty:
                yield pd.DataFrame({"__rid": [], id_col: []})
                continue

            pdf = pdf.sort_values([time_col, "__rid"]).reset_index(drop=True)
            ev = []
            gid = 1
            anchor = pdf.loc[0, time_col]
            ev.append(gid)
            for v in pdf[time_col].iloc[1:]:
                if abs(v - anchor) <= threshold_ms:
                    ev.append(gid)
                else:
                    gid += 1
                    anchor = v  # アンカー更新
                    ev.append(gid)
            out = pd.DataFrame({"__rid": pdf["__rid"], id_col: ev})
            yield out

    df_ids = df_nonnull.mapInPandas(_assign_event_id, schema=schema_ids)

    # 4) 元DFへ戻す（t0_ms が NULL の行は event_id NULL のまま）
    df_out = (
        df1.join(df_ids, on="__rid", how="left")
           .drop("__rid")
    )
    return df_out

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

    pulse_sum = F.aggregate(
        F.col("pulse"),
        F.lit(0.0),
        lambda acc, x: acc + F.coalesce(x.cast("double"), F.lit(0.0))
    ).alias("charge")

    df_aug = (df
        .select("chip", "channel", "pulse", "t0_ms")
        .withColumn("samidare_id", F.col("chip") * F.lit(32) + F.col("channel"))
        .withColumn("maxsample", F.array_max("pulse").cast("long")) 
        .withColumn("charge", pulse_sum)  
    )

    if 1:

        freq_chip         = (df_aug.groupBy("chip").count().orderBy("chip"))
        freq_channel      = (df_aug.groupBy("channel").count().orderBy("channel"))
        freq_samidare_id = (df_aug.groupBy("samidare_id").count().orderBy(F.desc("count")))
        freq_maxsample    = (df_aug.where(F.col("maxsample").isNotNull()).groupBy("maxsample").count().orderBy(F.desc("count")))

        df_histo_chip = freq_chip.toPandas()
        df_histo_chan = freq_channel.toPandas()
        df_histo_sami = freq_samidare_id.toPandas()
        df_histo_maxh = freq_maxsample.toPandas()

        data1 = [df_histo_chip["chip"], df_histo_chip["count"]] 
        data2 = [df_histo_chan["channel"], df_histo_chan["count"]] 
        data3 = [df_histo_sami["samidare_id"], df_histo_sami["count"]] 
        data4 = [df_histo_maxh["maxsample"], df_histo_maxh["count"]] 

        fig = make_subplots(rows=2, cols=2, vertical_spacing=0.15, horizontal_spacing=0.1,
            subplot_titles=( "sampa chip", "sampa channel", "samidare id", "maximum sample value" ))

        pau.add_sub_plot(fig,1,1,'sparck-hist',[*data1],['sampa chip ID','Counts'])
        pau.add_sub_plot(fig,1,2,'sparck-hist',[*data2],['sampa channel','Counts'])
        pau.add_sub_plot(fig,2,1,'sparck-hist',[*data3],['samidare id','Counts'])
        pau.add_sub_plot(fig,2,2,'sparck-hist',[*data4],['max sample','Counts'])

        fig.update_layout( height=800, width=1000, showlegend=False,title_text=f"{input_finename}")
        fig.show()

        df_ts = df_aug.select("t0_ms", "maxsample").where(F.col("t0_ms").isNotNull() & F.col("maxsample").isNotNull())
        pdf = df_ts.toPandas()
        data5 = [pdf["t0_ms"].tolist(), pdf["maxsample"].tolist()]

        fig = make_subplots(rows=1, cols=1, vertical_spacing=0.15, horizontal_spacing=0.1, subplot_titles=( "test" ))
        pau.add_sub_plot(fig,1,1,'plot',[*data5],['timestamp','max sample'],[100,100], [False,False,False])
        fig.update_layout( height=800, width=1000, showlegend=False,title_text=f"{input_finename}")
        fig.show()

    df_ev = add_event_id_anchor(df_aug, time_col="t0_ms", threshold_ms=100.0, id_col="event_id")
    df_ev.show(100)

if __name__ == '__main__':
    main()

