"""!
@file pyspark_analusis_test.py
@version 1
@author FumiHubCNS
@date 2025-07-30T15:40:42+09:00
@brief template text
"""
import argparse
import pathlib
import samidare_lib.padinfo as pinfo
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from collections import defaultdict
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from collections import Counter
import matplotlib.pyplot as plt

this_file_path = pathlib.Path(__file__).parent

BASE = "/Users/fendo/Work/Data/root/sampa-minitpc-test/output/"
FILE = "20250920_jpspos_30-160_64-16_395_455_001_00000"
DATA = BASE+FILE+".parquet"
MAPFILEPATH = "/Users/fendo/Work/Program/uv-python/samidare-util/prm/minitpc-samidare-test-at-ribf-2025-07/minitpc.map"

def get_id_from_mapdf(mapdf, sampaNo=2, sampaID=4, label='gid'):
    matched = mapdf.loc[(mapdf['sampaNo'] == sampaNo) & (mapdf['sampaID'] == sampaID), label]
    gid = matched.iloc[0] if not matched.empty else None
    return gid

def get_any_from_mapdf(mapdf, refLabel='sampaNo', refIDID=4):
    matched = mapdf[(mapdf[refLabel] == refIDID)]
    return matched

def main():

    parser = argparse.ArgumentParser()

    # parser.add_argument("-i", "--input", help="input file path", type=str, default="/Users/fendo/Work/Data/root/sampa-minitpc-test/test_TPC_001.root")
    #parser.add_argument("-i", "--input", help="input file path", type=str, default="/Users/fendo/Work/Data/root/sampa-minitpc-test/test_TPC_1kHz_001.root")  
    parser.add_argument("-n", "--maxn", help="event number for loading data using ak", type=int, default=-1)  
    parser.add_argument("-f", "--plotly-flag", help="sample flag", action="store_true")
    parser.add_argument("-fh", "--hitpatt-flag", help="sample flag", action="store_true")

    args = parser.parse_args()

    limit_number =  args.maxn
    pulse_check_flag = args.plotly_flag
    hitpatt_check_flag = args.hitpatt_flag

    offset = -3.031088913245535
    pad1 = pinfo.get_tpc_info(offset+45)
    pad2 = pinfo.get_tpc_info(offset+136.5)
    tpcs = pinfo.marge_padinfos(pad1,pad2)

    mapdf = pd.read_csv(MAPFILEPATH)
    mapdf = mapdf[pd.to_numeric(mapdf['tpcID'], errors='coerce').notna()]
    mapdf = mapdf[pd.to_numeric(mapdf['padID'], errors='coerce').notna()]

    mapdf['tpcID'] = mapdf['tpcID'].astype(int)
    mapdf['padID'] = mapdf['padID'].astype(int)
    mapdf['sampaNo'] = mapdf['sampaNo'].astype(int)
    mapdf['sampaID'] = mapdf['sampaID'].astype(int)
    mapdf['samidareID'] = mapdf['samidareID'].astype(int)
    mapdf = mapdf.reset_index(drop=True)
    mapdf['gid'] = (mapdf['tpcID'] - 1) * 60 + mapdf['padID'] - 1

    spark = SparkSession.builder.appName("MySparkApp").getOrCreate()
    data = spark.read.parquet(DATA)
    print(f"Full row count: {data.count()}")
    
    # data.show(150, truncate=False)

    if limit_number > 0:
        data = data.limit(limit_number)
    rows = data.collect()

    plotnumflgs = [0] * 200

    pltrows=11
    pltcols=12
    titles = []
    hitpatt=[]

    for i in range (pltrows*pltcols):
        ldata = get_any_from_mapdf(mapdf, 'samidareID',int(i+1))
        if not ldata.empty:
            titles.append(f"{i}: ({int(ldata.sampaNo.iloc[0])},{int(ldata.sampaID.iloc[0])}) [{int(ldata.samidareID.iloc[0])}]")
        else:
            titles.append(f"{i}: (N/A, N/A) [N/A]")
    
    if pulse_check_flag:
        fig = make_subplots(rows=pltrows, cols=pltcols, subplot_titles=titles)
          
    for row in rows:
        chip = row.chip
        ch = row.channel 
        waveform = row.samples
        timestamp = row.timestamp
        samid = get_id_from_mapdf(mapdf,chip,ch,'samidareID')
        gid = get_id_from_mapdf(mapdf,chip,ch,'gid')
        y = None

        if len(waveform)>50 and len(waveform)<65:
            baseline = np.mean(waveform[0:10]) 
            y = waveform - baseline
            x = list(range(len(waveform)))
            max_index = np.argmax(y)

        if samid is not None:
            pltrow = ((samid-1) // pltcols) + 1
            pltcol = ((samid-1) % pltcols) + 1

            if y is not None:
                if max(y) >50:
                    hitpatt.append(gid)
                    # print(f"gid: {gid}, インデックス: {max_index}, 最大値: {y[max_index]:10.5f}")
                    if plotnumflgs[samid] < 10:
                        if pulse_check_flag:
                            fig.add_trace(go.Scatter(x=x, y=y), row=pltrow, col=pltcol)
                    plotnumflgs[samid] += 1

    #                 plt.plot(x,y,label=f"({chip},{ch}), samidare-id: {samid}, tpc-id: {gid}")
    
    # plt.legend()
    # plt.show()

    if pulse_check_flag:
        fig.update_layout(title = f"Pulse Shape (self-triger mode) fumi decoder. subplot title : figure id (chip, channel) [samidare id]", showlegend=False)
        fig.update_annotations(font=dict(family="Helvetica", size=8))     
        fig.update_yaxes(range=[-200, 1000])#, row=row, col=col)
        fig.show()

    if hitpatt_check_flag:
        unique_lst = list(set(hitpatt))
        tpcs.show_pads(ref=unique_lst)

if __name__ == "__main__":
    main()
