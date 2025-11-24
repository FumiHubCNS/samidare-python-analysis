"""!
@file pyspark_hit_pattern.py
@version 1
@author FumiHubCNS
@date 2025-08-22T12:04:55+09:00
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

def fit_line(x, z):
    x = np.array(x)
    z = np.array(z)

    # 多項式フィット（1次）
    a, b = np.polyfit(z, x, 1)

    return a, b

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
    fileinfo = config["fileinfo"]
    tpc_map = analysinfo["tpc_mapfile"]
    input_path = fileinfo["base_output_path"] + "/" + fileinfo["input_file_name"] + "_00000_pulse.parquet"

    offset = -3.031088913245535
    pad1 = padinfo.get_tpc_info(offset+45)
    pad2 = padinfo.get_tpc_info(offset+136.5,False)
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

    spark = SparkSession.builder.appName("IterativeParquetReader").getOrCreate()
    data = spark.read.parquet(input_path).orderBy("timestamp")
    print(f"Full row count: {data.count()}")

    samids = []
    gids = []
    vgids = []
    tss = []
    chips = []
    chip_ch = []
    qs = []
    mss = []

    nmax  = -1
    inum = 0
    hitpatt = []
    hitpatt_qs = []
    hitpatt_up_xs = []
    hitpatt_up_ys = []
    hitpatt_up_zs = []
    hitpatt_dn_xs = []
    hitpatt_dn_ys = []
    hitpatt_dn_zs = []
    hitpatt_up_qs = []
    hitpatt_dn_qs =[]
    hitpatt_valid_id = []
    hitpatt_valid_qs = []
    testid = []
    spul_y = []
    spul_x = []
    evtid = []
    
    chip0sum = []
    chip1sum = []
    chip2sum = []
    chip3sum = []

    rows, cols = 4, 32
    chip_channel_charges = [[[] for _ in range(cols)] for _ in range(rows)]

    check_map = [0] * 128

    tpc_chip = []
    tpc_channel = []
    for i in range(120):
        tpc_ref_sampa_chip = get_any_from_mapdf_using_ref(mapdf,refLabel='gid',refID=i,label='sampaNo')
        tpc_ref_sampa_channel = get_any_from_mapdf_using_ref(mapdf,refLabel='gid',refID=i,label='sampaID')
        tpc_ref_samidare_id = get_any_from_mapdf_using_ref(mapdf,refLabel='gid',refID=i,label='samidareID')
        tpc_chip.append(tpc_ref_sampa_chip)
        tpc_channel.append(tpc_ref_sampa_channel)
        print(f"tpc id: {i}, samidare id: {tpc_ref_samidare_id}, sampa: ({tpc_ref_sampa_chip}, {tpc_ref_sampa_channel})")

    if 0:
        cehck_list = tpc_chip
        bins, colors = catview.get_color_list(cehck_list, cmap_name="rainbow", fmt="hex")
        color_array  = catview.get_color_array(cehck_list,bins,colors)
        tpcs.show_pads(check_id=True, check_size=3, plot_type='map',color_map=color_array, check_data=tpc_channel)

    if 0:
        for i in range(len(tpcs.ids)):
            print(f"{tpcs.ids[i]}, {tpcs.centers[i]}")

    if 0:
        for i in range(4):
            for j in range(32):
                samid = get_id_from_mapdf(mapdf,i, j,'samidareID')
                gid = get_id_from_mapdf(mapdf,i, j,'gid')
                tpcid = get_id_from_mapdf(mapdf,i, j,'tpcID')
                padid = get_id_from_mapdf(mapdf,i, j,'padID')
                # if gid is not pd.NA:
                #     # print(f"({i}, {j}) -> samidare iD: {samid}, Beam TPC ID: {gid}, ({tpcid}, {padid})")
                print(f"{i}, {j}, {samid}, {gid}, {tpcid}, {padid}")

    if 1:
        for row in data.toLocalIterator():
            evt = row["event_id"]
            chi = row["chip" ]
            chn = row["channel"] 
            tsp = row["timestamp"] 
            pul = row["pulse"]
            mul = row["multiplicity"]
            mse = row["max_sample"]
            cha = row["charge"]
            tch =row["total_charge"]
            msi =row["time"]
            hits = []
            vqs =[]
            up_xs = []
            up_ys = []
            up_zs = []
            dn_xs = []
            dn_ys = []
            dn_zs = []
            up_qs = []
            dn_qs = []

            mints = min(tsp)

            if inum < 20:
                print(f"evt: {inum}, timestamp: {tsp[0]}")

            qs0=0
            qs1=0
            qs2=0
            qs3=0

            for ii in range(4):
                for jj in range(32):
                    chip_channel_charges[ii][jj].append(0)

            for i in range(len(chi)):
                # samid = get_id_from_mapdf(mapdf,chi[i],chn[i],'samidareID')
                # gid = get_id_from_mapdf(mapdf,chi[i],chn[i],'gid')
                samid = chi[i] * 32 + chn[i]
                gid = get_any_from_mapdf_using_ref(mapdf,refID=samid)
                gid = gid if gid >=0 else pd.NA

                norm_pulse = list(np.array(pul[i])/max(pul[i]))
                norm_index = pau.get_1dindex_array(norm_pulse)
                spul_y.append(norm_pulse)
                spul_x.append(norm_index)

                samids.append(samid)
                gids.append(gid)
                tss.append(tsp[i])
                chips.append(chi[i])
                chip_ch.append(chn[i])
                qs.append(cha[i])
                mss.append(mse[i])
                evtid.append(inum)

                testid.append( 32*chi[i] + chn[i] )

                chip_channel_charges[chi[i]][chn[i]][inum] = cha[i]

                if  chi[i] == 0 and chn[i] <= 15:
                    qs0 = qs0 + cha[i]
                
                if  chi[i] == 1 and chn[i] <= 15:
                    qs1 = qs1 + cha[i]

                if  chi[i] == 2 and chn[i] <= 15:
                    qs2 = qs2 + cha[i]

                if  chi[i] == 3 and chn[i] <= 15:
                    qs3 = qs3 + cha[i]

                if gid is not pd.NA:
                    hits.append(gid)
                    vgids.append(gid)
                    vqs.append(mse[i])

                    if gid<60:
                        up_xs.append(tpcs.centers[gid][0])
                        up_ys.append(tpcs.centers[gid][1])
                        up_zs.append(tpcs.centers[gid][2])
                        up_qs.append(mse[i])

                    else:
                        dn_xs.append(tpcs.centers[gid][0])
                        dn_ys.append(tpcs.centers[gid][1])
                        dn_zs.append(tpcs.centers[gid][2])
                        dn_qs.append(mse[i])

                check_map[samid]=gid
            
            hitpatt.append(hits)
            hitpatt_qs.append(vqs)

            threshold = 60
            count = sum(1 for n in hits if n >= threshold)

            if  count > 5:
                hitpatt_valid_id.append(hits)
                hitpatt_valid_qs.append(vqs)
                hitpatt_up_xs.append(up_xs)
                hitpatt_up_ys.append(up_ys)
                hitpatt_up_zs.append(up_zs)
                hitpatt_dn_xs.append(dn_xs)
                hitpatt_dn_ys.append(dn_ys)
                hitpatt_dn_zs.append(dn_zs)
                hitpatt_up_qs.append(up_qs)
                hitpatt_dn_qs.append(dn_qs)

            if qs0 > 0 and  qs1 > 0 and qs2 > 0 and  qs3 > 0 : 
                chip0sum.append(qs0)
                chip1sum.append(qs1)
                chip2sum.append(qs2)
                chip3sum.append(qs3)

            if nmax == inum:
                break

            inum = inum + 1

        if 0:
            bin_edges = np.arange(0, 128 + 1, 1)
            counts1, bin_edges1 = np.histogram(samids, bins=bin_edges)
            print("SAMIDARE ID（counts）:", counts1)
            
            counts2, bin_edge2 = np.histogram(vgids, bins=bin_edges)
            print("Beam TPC ID（counts）:", counts2)

            for i in range(len(counts1)):
                print(f"{i} {counts1[i]} {i} {counts2[i]}")
            
            for i in range(len(check_map)):
                print(f"{i} {check_map[i]}")

        if 1:
            print(f"summary : num of evt {inum}")

        if 0: 
            for i in range(len(hitpatt)):
                savepath = str(this_file_path.parent.parent.parent / f"figs/{i:5d}.png")
                unique_lst = hitpatt[i]
                unique_qlst = hitpatt_qs[i]
                # tpcs.show_pads(ref=unique_lst)

                threshold = 60
                count = sum(1 for n in unique_lst if n >= threshold)

                if  count > 5:
                    q_lst = [0] * len(tpcs.ids)
                    for j in range(len(unique_lst)):
                        q_lst[int(unique_lst[j])] = unique_qlst[j]
                        
                    cehck_list = q_lst
                    bins, colors = catview.get_color_list(cehck_list, cmap_name="viridis", fmt="hex")
                    color_array  = catview.get_color_array(cehck_list,bins,colors)
                    tpcs.show_pads(plot_type='map', color_map=color_array)
            
        if 0:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "samidare id",
                    "global id",
                    "sampa",
                    "channel"
                ),
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )

            pau.add_sub_plot(fig,1,1,'1d',[samids],['Samidare ID','Counts'],[128],[0,128,1])
            pau.add_sub_plot(fig,1,2,'1d',[vgids],['Beam TPC ID','Counts'],[128],[0,128,1])
            pau.add_sub_plot(fig,2,1,'1d',[chips],['Chip Number','Counts'],[4])
            pau.add_sub_plot(fig,2,2,'1d',[chip_ch],['Channel ID','Counts'],[32])

            fig.update_layout(
                height=800,
                width=1000,
                title_text="Distributions of Multiplicity and Sampling Counts",
                showlegend=False
            )
            fig.show()

        if 0:
            spul_x_1d = pau.get_1darray_from_2darray(spul_x)
            spul_y_1d = pau.get_1darray_from_2darray(spul_y)
            
            df = pd.DataFrame({'chips': chips, 'channels': chip_ch, 'q': qs, 'max_sample': mss, 'gid': testid})
            dfp = pd.DataFrame({'pulse': spul_y_1d, 'index': spul_x_1d})

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "samidare id",
                    "global id",
                    "sampa",
                    "channel",
                ),
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )

            x = df["channels"].to_numpy()
            y = df["chips"].to_numpy()
            pau.add_sub_plot(fig,1,1,'2d',[x,y],['chanel','chip'],[32, 4])

            x = df["gid"].to_numpy()
            y = df["q"].to_numpy()
            pau.add_sub_plot(fig,1,2,'2d',[x,y],['id','charge'],[128, 50])

            x = dfp["index"].to_numpy()
            y = dfp["pulse"].to_numpy()
            pau.add_sub_plot(fig,2,1,'2d',[x,y],['index','normalized pulse'],[30, 50])

            pau.add_sub_plot(fig,2,2,'scatter',[tss],['timestamp','counts'])

            fig.update_layout(
                height=1200,
                width=1600,
                title_text="Channel Data (SAMIDARE original ID  and Beam TPC ID [user defined] )",
                showlegend=False
            )
            fig.show()


        if 0:            
            df = pd.DataFrame({'Q0': chip0sum,'Q1': chip1sum,'Q2': chip2sum,'Q3': chip3sum})

            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=(
                    "chip charge 0-1",
                    "chip charge 0-2",
                    "chip charge 0-3",
                    "chip charge 1-2",
                    "chip charge 1-3",
                    "chip charge 2-3",
                ),
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )

            x = df["Q0"].to_numpy()
            y = df["Q1"].to_numpy()
            pau.add_sub_plot(fig,1,1,'2d',[x,y],['Q0','Q1'],[50, 50],[False,False,False],[1,14e3],[1,14e3],True)
            

            x = df["Q0"].to_numpy()
            y = df["Q2"].to_numpy()
            pau.add_sub_plot(fig,1,2,'2d',[x,y],['Q0','Q2'],[50, 50],[False,False,False],[1,14e3],[1,14e3],True)

            x = df["Q0"].to_numpy()
            y = df["Q3"].to_numpy()
            pau.add_sub_plot(fig,1,3,'2d',[x,y],['Q0','Q3'],[50, 50],[False,False,False],[1,14e3],[1,14e3],True)

            x = df["Q1"].to_numpy()
            y = df["Q2"].to_numpy()
            pau.add_sub_plot(fig,2,1,'2d',[x,y],['Q1','Q2'],[50, 50],[False,False,False],[1,14e3],[1,14e3],True)

            x = df["Q1"].to_numpy()
            y = df["Q3"].to_numpy()
            pau.add_sub_plot(fig,2,2,'2d',[x,y],['Q1','Q3'],[50, 50],[False,False,False],[1,14e3],[1,14e3],True)

            x = df["Q2"].to_numpy()
            y = df["Q3"].to_numpy()
            pau.add_sub_plot(fig,2,3,'2d',[x,y],['Q2','Q3'],[50, 50],[False,False,False],[1,14e3],[1,14e3],True)

            fig.update_layout(
                height=1200,
                width=1600,
                title_text="Channel Data (SAMIDARE original ID  and Beam TPC ID [user defined] )",
                showlegend=False
            )
            fig.show()

        if 0:
            chipchannel_data = np.array(chip_channel_charges)

            xchip = 2
            xchannel = 0
            ychip = 3
            ychannel1 = 0
            ychannel2 = 1
            ychannel3 = 2
            ychannel4 = 3
            ychannel5 = 12
            ychannel6 = 13
            ychannel7 = 14
            ychannel8 = 15

            fig = make_subplots(
                rows=3, cols=3,
                subplot_titles=(
                    f"q({xchip}, {xchannel}) vs q({ychip}, {ychannel1})",
                    f"q({xchip}, {xchannel}) vs q({ychip}, {ychannel2})",
                    f"q({xchip}, {xchannel}) vs q({ychip}, {ychannel3})",
                    f"q({xchip}, {xchannel}) vs q({ychip}, {ychannel4})",
                    f"q({xchip}, {xchannel}) vs q({ychip}, {ychannel5})",
                    f"q({xchip}, {xchannel}) vs q({ychip}, {ychannel6})",
                    f"q({xchip}, {xchannel}) vs q({ychip}, {ychannel7})",
                    f"q({xchip}, {xchannel}) vs q({ychip}, {ychannel8})"
                ),
                vertical_spacing=0.15,
                horizontal_spacing=0.125
            )

            pau.add_sub_plot(fig,1,1,'2d',[chipchannel_data[xchip][xchannel],chipchannel_data[ychip][ychannel1]],[f"({xchip},{xchannel})",f"({ychip},{ychannel1})"],[50, 50],[False,False,False],[1,4000],[1,4000],True)
            pau.add_sub_plot(fig,1,2,'2d',[chipchannel_data[xchip][xchannel],chipchannel_data[ychip][ychannel2]],[f"({xchip},{xchannel})",f"({ychip},{ychannel2})"],[50, 50],[False,False,False],[1,4000],[1,4000],True)
            pau.add_sub_plot(fig,1,3,'2d',[chipchannel_data[xchip][xchannel],chipchannel_data[ychip][ychannel3]],[f"({xchip},{xchannel})",f"({ychip},{ychannel3})"],[50, 50],[False,False,False],[1,4000],[1,4000],True)
            pau.add_sub_plot(fig,2,1,'2d',[chipchannel_data[xchip][xchannel],chipchannel_data[ychip][ychannel4]],[f"({xchip},{xchannel})",f"({ychip},{ychannel4})"],[50, 50],[False,False,False],[1,4000],[1,4000],True)
            pau.add_sub_plot(fig,2,2,'2d',[chipchannel_data[xchip][xchannel],chipchannel_data[ychip][ychannel5]],[f"({xchip},{xchannel})",f"({ychip},{ychannel5})"],[50, 50],[False,False,False],[1,4000],[1,4000],True)
            pau.add_sub_plot(fig,2,3,'2d',[chipchannel_data[xchip][xchannel],chipchannel_data[ychip][ychannel6]],[f"({xchip},{xchannel})",f"({ychip},{ychannel6})"],[50, 50],[False,False,False],[1,4000],[1,4000],True)
            pau.add_sub_plot(fig,3,1,'2d',[chipchannel_data[xchip][xchannel],chipchannel_data[ychip][ychannel7]],[f"({xchip},{xchannel})",f"({ychip},{ychannel7})"],[50, 50],[False,False,False],[1,4000],[1,4000],True)
            pau.add_sub_plot(fig,3,2,'2d',[chipchannel_data[xchip][xchannel],chipchannel_data[ychip][ychannel8]],[f"({xchip},{xchannel})",f"({ychip},{ychannel8})"],[50, 50],[False,False,False],[1,4000],[1,4000],True)

            for trace in fig.data:
                if isinstance(trace, go.Heatmap):
                    # trace.xaxis の値は "x", "x2", "x3" など
                    xaxis = "xaxis" if trace.xaxis == "x" else "xaxis" + trace.xaxis[1:]
                    yaxis = "yaxis" if trace.yaxis == "y" else "yaxis" + trace.yaxis[1:]

                    # 各サブプロットの domain を取得
                    xa = fig.layout[xaxis].domain
                    ya = fig.layout[yaxis].domain

                    # カラーバーをサブプロットの右横に配置
                    trace.update(colorbar=dict(
                        thickness=20,            # 横幅（デフォルトは30くらい）
                        thicknessmode="pixels",  # ピクセル単位で指定
                        x=xa[1] + 0.01,   # サブプロットの右端の少し外
                        y=(ya[0] + ya[1]) / 2,  # 縦位置は中央
                        len=ya[1] - ya[0]       # サブプロットの高さに合わせる
                    ))


            fig.update_layout(
                height=950,
                width=1400,
                title_text=f"Charge Charge Correlation ({xchip}, {xchannel}) vs Chip {ychip}",
                showlegend=False
            )

            fig.show()
    


        if 0:
            chipchannel_data = np.array(chip_channel_charges)

            xchip = 2
            xchannel = 15
            ychip = 0

            base_titles = []
            for i in range(16):
                base_titles.append(f"q({xchip}, {xchannel}) vs q({ychip}, {int(i)})")

            fig = make_subplots(
                rows=4, cols=4,
                subplot_titles=base_titles,
                vertical_spacing=0.15,
                horizontal_spacing=0.125
            )

            k = 0
            for i in range(4):
                for j in range(4):
                    pau.add_sub_plot(fig,i+1,j+1,'2d',[chipchannel_data[xchip][xchannel],chipchannel_data[ychip][k]],[f"({xchip},{xchannel})",f"({ychip},{k})"],[50, 50],[False,False,False],[1,4000],[1,4000],False)
                    k = k + 1
                    
 
            for trace in fig.data:
                if isinstance(trace, go.Heatmap):
                    # trace.xaxis の値は "x", "x2", "x3" など
                    xaxis = "xaxis" if trace.xaxis == "x" else "xaxis" + trace.xaxis[1:]
                    yaxis = "yaxis" if trace.yaxis == "y" else "yaxis" + trace.yaxis[1:]

                    # 各サブプロットの domain を取得
                    xa = fig.layout[xaxis].domain
                    ya = fig.layout[yaxis].domain

                    # カラーバーをサブプロットの右横に配置
                    trace.update(colorbar=dict(
                        thickness=20,            # 横幅（デフォルトは30くらい）
                        thicknessmode="pixels",  # ピクセル単位で指定
                        x=xa[1] + 0.01,   # サブプロットの右端の少し外
                        y=(ya[0] + ya[1]) / 2,  # 縦位置は中央
                        len=ya[1] - ya[0]       # サブプロットの高さに合わせる
                    ))


            fig.update_layout(
                height=950,
                width=1400,
                title_text=f"Charge Charge Correlation ({xchip}, {xchannel}) vs Chip {ychip}",
                showlegend=False
            )

            fig.update_annotations(font=dict(size=10)) 
            fig.show()


        if 1:

            up_xs_1d = np.array(pau.get_1darray_from_2darray(hitpatt_up_xs))
            dn_xs_1d = np.array(pau.get_1darray_from_2darray(hitpatt_dn_xs))

            mean_up_xs_1d = []
            mean_dn_xs_1d = []
            sumq_up_xs_1d = []
            sumq_dn_xs_1d = []

            track_up_as_1d = []
            track_dn_as_1d = []
            track_up_bs_1d = []
            track_dn_bs_1d = []
            track_al_as_1d = []
            track_al_bs_1d = []


            for i in range(len(hitpatt_up_xs)):
                # mean_up_xs_1d.append(np.mean(hitpatt_up_xs[i]))
                sumq_up_xs_1d.append(np.sum(hitpatt_up_qs[i]))
                values = hitpatt_up_xs[i]
                weights = hitpatt_up_qs[i]
                weighted_avg = sum(v * w for v, w in zip(values, weights)) / sum(weights)
                mean_up_xs_1d.append(weighted_avg)

                a, b = fit_line(hitpatt_up_xs[i], hitpatt_up_zs[i])

                track_up_as_1d.append(a)
                track_up_bs_1d.append(b)

                # mean_dn_xs_1d.append(np.mean(hitpatt_dn_xs[i]))
                sumq_dn_xs_1d.append(np.sum(hitpatt_dn_qs[i]))
                values = hitpatt_dn_xs[i]
                weights = hitpatt_dn_qs[i]
                weighted_avg = sum(v * w for v, w in zip(values, weights)) / sum(weights)
                mean_dn_xs_1d.append(weighted_avg)

                a, b = fit_line(hitpatt_dn_xs[i], hitpatt_dn_zs[i])

                track_dn_as_1d.append(a)
                track_dn_bs_1d.append(b)

                total_xs = hitpatt_up_xs[i] + hitpatt_dn_xs[i]
                total_zs = hitpatt_up_zs[i] + hitpatt_dn_zs[i]

                a, b = fit_line(total_xs, total_zs)

                track_al_as_1d.append(a)
                track_al_bs_1d.append(b)

            if 0:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        "samidare id",
                        "global id",
                        "sampa",
                        "channel",
                    ),
                    vertical_spacing=0.15,
                    horizontal_spacing=0.1
                )

                pau.add_sub_plot(fig,1,1,'2d',[mean_up_xs_1d,mean_dn_xs_1d],[f"maxsample weighted x position at upstream",f"maxsample weighted x position at downstream"],[100, 100],[False,False,False],[-20,20],[-20,20],True)
                pau.add_sub_plot(fig,1,2,'2d',[sumq_up_xs_1d,sumq_dn_xs_1d],[f"sum of q at upstream",f"sum of q at downstream"],[100, 100],[False,False,False],[0,8000],[0,12e3],True)
                pau.add_sub_plot(fig,2,1,'2d',[track_up_as_1d,track_dn_as_1d],[f"a at upstream",f"a at downstream"],[100, 100], [False,False,False],[-3,3],[-3,3],True)
                pau.add_sub_plot(fig,2,2,'2d',[track_up_bs_1d,track_dn_bs_1d],[f"b at upstream",f"b at downstream"],[100, 100], [False,False,False],[-150,150],[-150,150],True)

                for trace in fig.data:
                    if isinstance(trace, go.Heatmap):
                        # trace.xaxis の値は "x", "x2", "x3" など
                        xaxis = "xaxis" if trace.xaxis == "x" else "xaxis" + trace.xaxis[1:]
                        yaxis = "yaxis" if trace.yaxis == "y" else "yaxis" + trace.yaxis[1:]

                        # 各サブプロットの domain を取得
                        xa = fig.layout[xaxis].domain
                        ya = fig.layout[yaxis].domain

                        # カラーバーをサブプロットの右横に配置
                        trace.update(colorbar=dict(
                            thickness=20,            # 横幅（デフォルトは30くらい）
                            thicknessmode="pixels",  # ピクセル単位で指定
                            x=xa[1] + 0.01,   # サブプロットの右端の少し外
                            y=(ya[0] + ya[1]) / 2,  # 縦位置は中央
                            len=ya[1] - ya[0]       # サブプロットの高さに合わせる
                        ))

                fig.update_layout(
                    height=950,
                    width=1400,
                    title_text="Channel Data (SAMIDARE original ID  and Beam TPC ID [user defined] )",
                    showlegend=False
                )
                fig.show()

            if 0:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        "samidare id",
                        "global id",
                        "sampa",
                        "channel",
                    ),
                    vertical_spacing=0.15,
                    horizontal_spacing=0.1
                )

                pau.add_sub_plot(fig,1,1,'1d',[up_xs_1d],['x position(up)','Counts'],xrange=[-20,20,1])
                pau.add_sub_plot(fig,1,2,'1d',[dn_xs_1d],['x position(down)','Counts'],xrange=[-20,20,1])
                pau.add_sub_plot(fig,2,1,'1d',[mean_up_xs_1d],['x position(up)','Counts'],xrange=[-20,20,1])
                pau.add_sub_plot(fig,2,2,'1d',[mean_dn_xs_1d],['x position(down)','Counts'],xrange=[-20,20,1])

                fig.update_layout(
                    height=950,
                    width=1400,
                    title_text="Channel Data (SAMIDARE original ID  and Beam TPC ID [user defined] )",
                    showlegend=False
                )

                fig.show()

            if 1: 
                for i in range(len(hitpatt_valid_id)):
                    tracks=[]
                    tracks.append(['line',[track_up_as_1d[i],track_up_bs_1d[i]],[35,145],[1,'red']])
                    tracks.append(['line',[track_dn_as_1d[i],track_dn_bs_1d[i]],[35,145],[1,'blue']])
                    tracks.append(['line',[track_al_as_1d[i],track_al_bs_1d[i]],[35,145],[1,'green']])

                    savepath = str(this_file_path.parent.parent.parent / f"figs/{i:5d}.png")
                    unique_lst = hitpatt_valid_id[i]
                    unique_qlst = hitpatt_valid_qs[i]

                    q_lst = [0] * len(tpcs.ids)
                    for j in range(len(unique_lst)):
                        q_lst[int(unique_lst[j])] = unique_qlst[j]
                        
                    cehck_list = q_lst
                    bins, colors = catview.get_color_list(cehck_list, cmap_name="viridis", fmt="hex")
                    color_array  = catview.get_color_array(cehck_list,bins,colors)
                    tpcs.show_pads(plot_type='map', color_map=color_array, tracks=tracks,xrange=[-20,20],block_flag=False)


if __name__ == '__main__':
    main()
