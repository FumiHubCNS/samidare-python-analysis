"""!
@file pyspark_check_raw_pulse.py
@version 1
@author FumiHubCNS
@date 2025-09-02T23:07:32+09:00
@brief template text
"""
import click
import pathlib
import datetime
import toml
import sys
from pyspark.sql import SparkSession
import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import toml
import pathlib
import sys
from collections import defaultdict
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

this_file_path = pathlib.Path(__file__).parent
sys.path.append(str(this_file_path.parent.parent.parent / "src"))

import samidare_lib.decoder.samidare_decorder_version2 as samidare_decoder
import samidare_lib.decoder.pyspark_pulse_analysis_version2 as pulse_analysis

def common_options(func):
    # @click.option('--name', '-n', default='World', show_default=True, help='Your name')
    @click.option('--date', '-d', type=click.DateTime(formats=["%Y-%m-%d"]), default=lambda: datetime.datetime.now(), show_default=lambda: datetime.datetime.now().strftime("%Y-%m-%d"), help="Date in YYYY-MM-DD format")
    @click.option('--verbose', '-v', is_flag=True, help='verbose flag')
    @click.option('--basesubtract', '-bs', is_flag=True, help='base line subtraction flag')
    @click.option('--nmax', '-n', type=int, help='maximum loop number',default=10000)
    @click.option('--eventflag', '-ef', is_flag=True, help='flag for plot pulse event block by event block')                                                                                               
    @click.option('--channelflag', '-cf', is_flag=True, help='flag for plot pulse with each channel')
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@common_options
def main(date, verbose,basesubtract, nmax, eventflag, channelflag):
    if verbose:
        click.echo(f"[VERBOSE MODE] date: {date.strftime('%Y-%m-%d')}")
    else:
        click.echo(f"(Date: {date.strftime('%Y-%m-%d')})")

    #### load decode structure and input path information
    toml_file_path = this_file_path  / "../../../parameters.toml"

    with open(toml_file_path, "r") as f:
        config = toml.load(f)

    format = config["format"]
    fileinfo = config["fileinfo"]
    analysinfo = config["analysis"]


    #### decode from binary file
    HEADER_MARKER = format["header_marker"]# = 0xafaf
    FOOTER_MARKER = format["footer_marker"]# = 0xfafa
    T1_MARKER = format["t1_marker"]# = 0xfffa
    T2_MARKER = format["t2_marker"]# = 0xfaaf
    T3_MARKER = format["t3_marker"]# = 0xaffa  
    HEADER_SIZE = format["header_size"]# = 4
    TN_SIZE = format["timestamp_marker_size"]# = 6
    FOOTER_SIZE = format["footer_size"]# = 2
    TIMESTAMP_SIZE = format["timestamp_size"]# = 6
    DATA_SIZE = format["data_size"]# = 40
    NUM_CHANNELS = format["num_channels"]# = 32
    MAX_NUM_SAMPLES = format["max_num_samples"]# = 64

    OUTPUTPATH = fileinfo["base_output_path"]
    BASE = fileinfo["base_input_path"] + "/" + fileinfo["input_file_dir"] + "/"
    FILE = fileinfo["input_file_name"] + ".bin"

    DATA = BASE+FILE
    NAME = FILE.removesuffix(".bin")

    output_paths = samidare_decoder.parse_binary_file_with_timestamp( DATA, HEADER_MARKER, FOOTER_MARKER, \
        HEADER_SIZE, TN_SIZE, FOOTER_SIZE, TIMESTAMP_SIZE, DATA_SIZE, NUM_CHANNELS, MAX_NUM_SAMPLES, OUTPUTPATH, NAME,
        nmax)

    ##### event building and pulse finding
    inum = 0

    spark = SparkSession.builder.appName("IterativeParquetReader").getOrCreate()
    df = spark.read.parquet(output_paths[0])

    ch_all_waveforms = defaultdict(list)

    plt.figure(figsize=(10, 6))

    for row in df.toLocalIterator():
        ts = row["timestamp"]
        sample = row["samples"]
        chip = row["chip"]
        channel = row["channel"]

        raw = sample
        id = channel + 32 * chip
        x = range(len(raw))
        counter = Counter(raw)
        most_common = counter.most_common(1)[0]  
        raw_signed = np.asarray(raw, dtype=np.int32) 
        baseline_signed = int(most_common[0])  
        y = raw_signed - baseline_signed if basesubtract else raw_signed
        x = np.arange(len(y))

        ch_all_waveforms[id].append(y)
    
        if inum == nmax:
            break
        
        if eventflag:
            if inum > 0 and inum % 504 == 0: 
                plt.title("Pulse Shape (all channels overlay)")
                plt.xlabel("Sample index")
                plt.ylabel("ADC - baseline")
                plt.show(block=False)
                plt.pause(1)
                plt.close() 
                plt.figure(figsize=(10, 6))
            else:
                plt.plot(x, y, lw=1, alpha=0.5, marker="o", markersize=2)
        
        inum += 1

    if channelflag:
        rows=11
        cols=12
        titles = []

        plotnumflgs = [0] * 200

        for i in range (rows*cols):
            titles.append(f"Ch:{i}")

        ref = ch_all_waveforms
            
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles)

        for ch, waveforms in ref.items():
            for wf in waveforms:
                row = (ch // cols) + 1
                col = (ch % cols) + 1
                x = np.arange(len(wf))
                if row > rows or col > cols:
                    print(f"⚠️ ch={ch} → (row={row}, col={col}) は範囲外。スキップします")
                    continue

                if plotnumflgs[ch] < 5:
                    fig.add_trace(go.Scatter(x= x, y= wf), row=row, col=col)
                plotnumflgs[ch] += 1

        fig.update_layout(title = f"Pulse Shape (self-triger mode)", showlegend=False)
        fig.update_annotations(font=dict(family="Helvetica", size=8))
        fig.update_yaxes(range=[-400, 600])
        fig.show()

if __name__ == '__main__':
    main()
