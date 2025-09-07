"""!
@file pyspark_check_conector_map.py
@version 1
@author FumiHubCNS
@date 2025-09-07T09:04:34+09:00
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

    df.show(20)



if __name__ == '__main__':
    main()

