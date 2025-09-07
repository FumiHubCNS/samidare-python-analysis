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
    # df.show(200)


    # col = "channel"  
    # freq_df = (
    #     df.select(F.col(col).cast("long").alias(col))
    #     .where(F.col(col).isNotNull())
    #     .groupBy(col).count()
    #     .orderBy(F.desc("count"))
    # )

    # pdf = freq_df.toPandas()

    # fig = go.Figure(go.Bar(x=pdf[col], y=pdf["count"]))
    # fig.update_layout(title=f"Frequency of {col}", xaxis_title=col, yaxis_title="count")
    # fig.show()


    df_aug = (df
        .select("chip", "channel", "pulse")
        .withColumn("chip_channel", F.col("chip") * F.lit(32) + F.col("channel"))
        .withColumn("pulse_max", F.array_max("pulse").cast("long"))  # 空配列は NULL になる
    )

    freq_chip         = (df_aug.groupBy("chip").count().orderBy("chip"))
    freq_channel      = (df_aug.groupBy("channel").count().orderBy("channel"))
    freq_chip_channel = (df_aug.groupBy("chip_channel").count().orderBy(F.desc("count")))
    freq_pulse_max    = (df_aug.where(F.col("pulse_max").isNotNull()).groupBy("pulse_max").count().orderBy(F.desc("count")))

    df_histo_chip = freq_chip.toPandas()
    df_histo_chan = freq_channel.toPandas()
    df_histo_sami = freq_chip_channel.toPandas()
    df_histo_maxh = freq_pulse_max.toPandas()

    data1 = [df_histo_chip["chip"], df_histo_chip["count"]] 
    data2 = [df_histo_chan["channel"], df_histo_chan["count"]] 
    data3 = [df_histo_sami["chip_channel"], df_histo_sami["count"]] 
    data4 = [df_histo_maxh["pulse_max"], df_histo_maxh["count"]] 

    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.15, horizontal_spacing=0.1,
        subplot_titles=( "sampa chip", "sampa channel", "samidare id", "maximum sample value" ))

    pau.add_sub_plot(fig,1,1,'sparck-hist',[*data1],['sampa chip ID','Counts'])
    pau.add_sub_plot(fig,1,2,'sparck-hist',[*data2],['sampa channel','Counts'])
    pau.add_sub_plot(fig,2,1,'sparck-hist',[*data3],['samidare id','Counts'])
    pau.add_sub_plot(fig,2,2,'sparck-hist',[*data4],['max sample','Counts'])

    fig.update_layout( height=800, width=1000, showlegend=False,
        title_text=f"{input_finename}")
        
    fig.show()


if __name__ == '__main__':
    main()

