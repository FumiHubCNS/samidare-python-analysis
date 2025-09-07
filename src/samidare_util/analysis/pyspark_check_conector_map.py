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
import toml
import sys
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
import catmlib.util.catmviewer as catview
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

this_file_path = pathlib.Path(__file__).parent
sys.path.append(str(this_file_path.parent.parent.parent / "src"))

import samidare_util.detector.padinfo as padinfo
import samidare_util.decoder.pyspark_pulse_analysis_version2 as pau

def get_id_from_mapdf(mapdf, sampaNo=2, sampaID=4, label='gid'):
    matched = mapdf.loc[(mapdf['sampaNo'] == sampaNo) & (mapdf['sampaID'] == sampaID), label]
    gid = matched.iloc[0] if not matched.empty else None
    return gid

def get_any_from_mapdf_using_ref(mapdf,refLabel='samidareID', refID=4, label='gid'):
    matched = mapdf.loc[(mapdf[refLabel] == refID), label]
    gid = matched.iloc[0] if not matched.empty else None
    return gid

def get_any_from_mapdf(mapdf, refLabel='sampaNo', refIDID=4):
    matched = mapdf[(mapdf[refLabel] == refIDID)]
    return matched

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

    analysinfo = config["analysis"]
    tpc_map = analysinfo["tpc_mapfile"]

    offset = -3.031088913245535
    pad1 = padinfo.get_tpc_info(offset-10)
    pad2 = padinfo.get_tpc_info(offset+10,False)
    tpcs = padinfo.marge_padinfos(pad1,pad2)

    mapdf = pd.read_csv(tpc_map)
    mapdf['padID'] = pd.to_numeric(mapdf['padID'], errors='coerce')
    mapdf['gid'] = pd.to_numeric(mapdf['gid'], errors='coerce')

    mapdf['tpcID'] = mapdf['tpcID'].astype(int)
    mapdf['padID'] = mapdf['padID'].fillna(-1).astype(int)
    mapdf['gid'] = mapdf['gid'].fillna(-1).astype(int)
    mapdf['sampaNo'] = mapdf['sampaNo'].astype(int)
    mapdf['sampaID'] = mapdf['sampaID'].astype(int)
    mapdf['samidareID'] = mapdf['samidareID'].astype(int)

    mapdf = mapdf.reset_index(drop=True)

    tpc_chip = []
    tpc_channel = []

    if 1:
        for i in range(120):
            tpc_ref_sampa_chip = get_any_from_mapdf_using_ref(mapdf,refLabel='gid',refID=i,label='sampaNo')
            tpc_ref_sampa_channel = get_any_from_mapdf_using_ref(mapdf,refLabel='gid',refID=i,label='sampaID')
            tpc_ref_samidare_id = get_any_from_mapdf_using_ref(mapdf,refLabel='gid',refID=i,label='samidareID')
            tpc_chip.append(tpc_ref_sampa_chip)
            tpc_channel.append(tpc_ref_sampa_channel)
            print(f"tpc id: {i}, samidare id: {tpc_ref_samidare_id}, sampa: ({tpc_ref_sampa_chip}, {tpc_ref_sampa_channel})")

    if 1:
        cehck_list = tpc_chip
        bins, colors = catview.get_color_list(cehck_list, cmap_name="rainbow", fmt="hex")
        color_array  = catview.get_color_array(cehck_list,bins,colors)
        tpcs.show_pads(check_id=True, check_size=13, plot_type='map',color_map=color_array, check_data=tpcs.ids)

    if 1:
        for i in range(len(tpcs.ids)):
            print(f"{tpcs.ids[i]}, {tpcs.centers[i]}")

    if 1:
        for i in range(4):
            for j in range(32):
                samid = get_id_from_mapdf(mapdf,i, j,'samidareID')
                gid = get_id_from_mapdf(mapdf,i, j,'gid')
                tpcid = get_id_from_mapdf(mapdf,i, j,'tpcID')
                padid = get_id_from_mapdf(mapdf,i, j,'padID')
                print(f"{i}, {j}, {samid}, {gid}, {tpcid}, {padid}")


if __name__ == '__main__':
    main()
