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

    pulse_sum = F.aggregate(
        F.col("pulse"),
        F.lit(0.0),
        lambda acc, x: acc + F.coalesce(x.cast("double"), F.lit(0.0))
    ).alias("charge")

    pos1 = F.array_position(F.col("pulse"), F.array_max(F.col("pulse")))
    idx = F.when(pos1 <= 0, F.lit(None)).otherwise(pos1.cast("int"))
    offset = (F.element_at(F.col("pulse_timestamp"), idx).cast("double") / F.lit(320000.0))
    ts_calc = F.coalesce(offset, F.lit(0.0))

    df_aug = (df
        .select("chip", "channel", "pulse", "t0_ms", "pulse_index", "pulse_timestamp")
        .withColumn("samidare_id", F.col("chip") * F.lit(32) + F.col("channel"))
        .withColumn("maxsample", F.array_max("pulse").cast("long"))
        .withColumn("charge", pulse_sum)
        .withColumn("ts_calc", ts_calc)
        .withColumn("ts_calcmod10", F.col("ts_calc") % F.lit(10))
        .filter(F.col("channel") == 15)
    )

    data1=[]
    data2=[]
    previous_ts = 0
    

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
    count = 0
    for row in df_aug.toLocalIterator():

        pulse_arr      = row['pulse']     
        index_arr      = row['pulse_index'] 
        channel_arr    = row['channel'] 
        current_ts     = row['t0_ms'] 


        if previous_ts > 0:

            data1.append(current_ts-previous_ts)
            data2.append(current_ts)
            print(current_ts-previous_ts)

        count += 1

        if 0:
            ax.plot(pulse_arr, 
                lw=1, alpha=0.5, marker="o", markersize=2, 
                label=f"ch{channel_arr}@{count}", 
                color=bd.color32(count,"brg")
            )

            if count == 8:
                ax.set(xlim=(0, 25), ylim=(-50, 250))
                ax.set_title(f"pulse @ chip{0}")     
                ax.set_xlabel("Sample index")
                ax.set_ylabel("Sample value - Base line(mode value)")   
                ax.legend(loc='upper right', ncol=3, fontsize=6)
                plt.show(block=True)  
                fig, ax = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
                count == 0

        if count < 200:
            # print(pulse_arr[0])
            print(current_ts%10)

        previous_ts = current_ts


    if 1:
        df_ts = df_aug.select("ts_calc", "maxsample").where(F.col("ts_calc").isNotNull() & F.col("maxsample").isNotNull())
        pdf = df_ts.toPandas()
        data5 = [pdf["ts_calc"].tolist(), pdf["maxsample"].tolist()]

        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.15, horizontal_spacing=0.1)
        pau.add_sub_plot(fig,1,1,'plot',[*data5],['timestamp[ms]','max sample'])
        pau.add_sub_plot(fig,2,1,'plot',[data2,data1],['timestamp[ms]','dt[ms]'])
        fig.update_xaxes(matches='x')
        fig.update_layout( height=800, width=1000, showlegend=False,title_text=f"{input_finename}")
        fig.show()


    if 0:	
        plt.hist(data1)#, bins=100,range=[9.999,10.001])
        plt.show()

  


if __name__ == '__main__':
    main()

