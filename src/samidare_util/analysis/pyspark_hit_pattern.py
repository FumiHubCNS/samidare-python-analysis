"""!
@file pyspark_hit_pattern.py
@version 1
@author FumiHubCNS
@date 2025-08-22T12:04:55+09:00
@brief template text
"""
import click
import pathlib
import toml
import sys
import pandas as pd
import numpy as np
import os 
import plotly.graph_objs as go

from pyspark.sql import SparkSession
from pyspark.sql import functions as F, types as T
from plotly.subplots import make_subplots
from math import isfinite

import catmlib.util.catmviewer as catview
import samidare_util.decoder.pyspark_pulse_analysis_version2 as pau
import samidare_util.analysis.savefig_util as saveutil

this_file_path = pathlib.Path(__file__).parent
sys.path.append(str(this_file_path.parent.parent.parent / "src"))

import samidare_util.detector.padinfo as padinfo

def common_options(func):   
    @click.option('--verbose', '-v', is_flag=True, help='verbose flag')
    
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def within_range(arr, low, high, *, inclusive="both", abs_value=False, finite_only=True):
    """
    arr中の値のうち [low, high]（inclusive='both' のとき）に入る要素だけ返す。
      inclusive: 'both' | 'left' | 'right' | 'neither'
      abs_value: True で |x| を範囲判定に使う
      finite_only: True で inf/-inf/NaN/None を除外
    """
    if low > high:
        low, high = high, low

    def ok(x):
        if not isinstance(x, (int, float)):
            return False
        if finite_only and not isfinite(x):
            return False
        v = abs(x) if abs_value else x
        if   inclusive == "both":    return low <= v <= high
        elif inclusive == "left":    return low <= v <  high
        elif inclusive == "right":   return low <  v <= high
        elif inclusive == "neither": return low <  v <  high
        else:  # 不正値は既定で両端含む
            return low <= v <= high

    return [x for x in arr if ok(x)]

def within_range_indices(arr, low, high, **kwargs):
    """範囲内に入る要素のインデックスを返す版"""
    vals = within_range(arr, low, high, **kwargs)
    # 同じ条件で再判定して index を返す（重複回避なら一度条件関数を外に出す実装でもOK）
    res = []
    if low > high:
        low, high = high, low
    abs_value = kwargs.get("abs_value", False)
    inclusive = kwargs.get("inclusive", "both")
    finite_only = kwargs.get("finite_only", True)

    def cond(x):
        if not isinstance(x, (int, float)): return False
        if finite_only and not isfinite(x): return False
        v = abs(x) if abs_value else x
        if   inclusive == "both":    return low <= v <= high
        elif inclusive == "left":    return low <= v <  high
        elif inclusive == "right":   return low <  v <= high
        elif inclusive == "neither": return low <  v <  high
        return low <= v <= high

    return [i for i, x in enumerate(arr) if cond(x)]

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

def get_pads():
    offset = -3.031088913245535
    pad1 = padinfo.get_tpc_info(offset+45)
    pad2 = padinfo.get_tpc_info(offset+136.5,False)
    tpcs = padinfo.marge_padinfos(pad1,pad2)
    return tpcs

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@common_options
def main(verbose):

    toml_file_path = this_file_path  / "../../../parameters.toml"

    with open(toml_file_path, "r") as f:
        config = toml.load(f)

    analysinfo = config["analysis"]
    fileinfo = config["fileinfo"]
    map_path = analysinfo["tpc_mapfile"]
    base_path = fileinfo["base_output_path"]  + "/" + fileinfo["input_file_name"] 
    input = base_path + "_phys.parquet"
    input_finename = os.path.basename(input)
    savebase = str(this_file_path.parent.parent.parent / "figs")

    schema_map = (
        T.StructType()
        .add("sampaNo",   T.IntegerType())
        .add("sampaID",   T.IntegerType())
        .add("samidareID",T.IntegerType())
        .add("tpcID",     T.IntegerType())
        .add("padID",     T.IntegerType())
        .add("gid",       T.StringType())  
    )

    spark = (
        SparkSession.builder
        .config("spark.driver.memory", "8g")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "128") 
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .config("spark.sql.files.maxPartitionBytes", 32 * 1024 * 1024)
        .getOrCreate()
    )

    df_map_raw = (spark.read
        .option("header", True)
        .schema(schema_map)
        .csv(map_path)
    )

    df_map = (df_map_raw
        .withColumn(
            "gid_mapped",
            F.when(F.col("gid") == F.lit("G"), F.lit(-1)).otherwise(F.col("gid").cast("int"))
        )
        .select(
            F.col("samidareID").cast("long").alias("map_samidare_id"),
            F.col("gid_mapped")
        )
    )

    df = spark.read.parquet(input)    

    df_gid = (df
        .join(F.broadcast(df_map), df["samidare_id"] == df_map["map_samidare_id"], "left")
        .drop("map_samidare_id")
        .withColumnRenamed("gid_mapped", "tpc_id")  
    )

    df_plot = (df_gid.filter(df_gid['tpc_id'] >= 0))


    if 0:
        data1 = (df_plot.groupBy("samidare_id").count().orderBy("samidare_id"))
        data2 = (df_plot.groupBy("tpc_id").count().orderBy("tpc_id"))

        df1 = data1.toPandas()
        df2 = data2.toPandas()


        fig = make_subplots(rows=1, cols=2, vertical_spacing=0.15, horizontal_spacing=0.1,
            subplot_titles=( "Samidare ID", "TPC ID" ))

        pau.add_sub_plot(fig,1,1,'sparck-hist',[df1["samidare_id"],df1["count"]],['Samidare ID','Counts'])
        pau.add_sub_plot(fig,1,2,'sparck-hist',[df2["tpc_id"],df2["count"]],['TPC ID','Counts'])

        fig.update_layout( height=800, width=1600, showlegend=False,title_text=f"{input_finename}")
        fig.show()
        # saveutil.save_plotly(fig, base_dir=savebase)


    qsum_arr = []
    qsum = 0
    hitpatt_id = []
    hitpatt_q = []

    ref_evt = 1
    figcount = 0
    uqx = 0 
    dqx = 0

    uqs = []
    dqs = []
    uposs = []
    dposs = []

    uqsum = []
    dqsum = []
    uxg = []
    dxg = []

    tpcs = get_pads()

    for row in df_plot.toLocalIterator():
        evtid  = row["event_id"]
        charge = row["charge"]
        tpcid = row["tpc_id"]

        if ref_evt == evtid:
            hitpatt_id.append(tpcid)
            hitpatt_q.append(charge)
            qsum += charge
        
        else:

            if 1:
                q_lst = [0] * len(tpcs.ids)
                for j in range(len(hitpatt_id)):
                    q_lst[int(hitpatt_id[j])] = hitpatt_id[j]
                    
                cehck_list = q_lst
                bins, colors = catview.get_color_list(cehck_list, cmap_name="ocean_r", fmt="hex")
                color_array  = catview.get_color_array(cehck_list,bins,colors)

                thr = 60
                count1 = sum(1 for v in hitpatt_id if v <  thr)
                count2 = sum(1 for v in hitpatt_id if v >= thr)

                up3_vidx = within_range_indices(hitpatt_id, 40, 59)
                sn1_vidx = within_range_indices(hitpatt_id, 60, 79)

                if len(up3_vidx) > 0 and len(sn1_vidx) > 0:
                    for uidx in up3_vidx:
                        pos = tpcs.centers[uidx]
                        q   = hitpatt_q[uidx]

                        uposs.append(pos[0])
                        uqs.append(q)

                    for didx in sn1_vidx:
                        pos = tpcs.centers[didx]
                        q   = hitpatt_q[didx]

                        dposs.append(pos[0])
                        dqs.append(q)
                    
                    uqsum.append(sum(uqs))
                    dqsum.append(sum(dqs))

                    for k in range(len(uqs)):
                        uqx += uqs[0] * uposs[0]

                    for k in range(len(dqs)):
                        dqx += dqs[0] * dposs[0]

                    uxg.append(uqx/sum(uqs))
                    dxg.append(dqx/sum(dqs))

                    uposs.clear()
                    uqs.clear()
                    dposs.clear()
                    dqs.clear()
                    uqx=0
                    dqx=0

                if count2 > 9 and count1 > 9:
                    if 0:
                        tpcs.show_pads(
                            plot_type='map', 
                            color_map=color_array, 
                            xrange=[-20,20],
                            block_flag=False,
                            savepath = None #f"{savebase}/20250907/{figcount:5d}.png"
                        )
                        figcount += 1
    
            hitpatt_id.clear()
            hitpatt_q.clear()
            qsum_arr.append(qsum)
            
            qsum = 0
            hitpatt_id.append(tpcid)
            hitpatt_q.append(charge)
            qsum += charge
            ref_evt = evtid

    if 0:
        fig = make_subplots(rows=1, cols=1, vertical_spacing=0.15, horizontal_spacing=0.1,
        subplot_titles=( "Total Charge", ))

        pau.add_sub_plot(fig,1,1,'1d',[qsum_arr],['Total Charge','Counts'],[200])

        fig.update_layout( height=800, width=1600, showlegend=False,title_text=f"{input_finename}")
        fig.show()
        # saveutil.save_plotly(fig, base_dir=savebase)

    if 1:
        fig = make_subplots(rows=2, cols=3, vertical_spacing=0.15, horizontal_spacing=0.1,
            subplot_titles=( 
                "upstream tpc total charge", 
                "downstream tpc total charge", 
                r"$q_{up} \,\mathrm{vs}\, q_{down}$", 
                "upstream tpc weighted x position", 
                "downstream tpc weighted x position", 
                r"$x_{g,up} \,\mathrm{vs}\, x_{g,down}$", 
            )
        )

        pau.add_sub_plot(fig,1,1,'1d',[uqsum],[r"$Q_{sum, up}$",'Counts'],[50])
        pau.add_sub_plot(fig,1,2,'1d',[dqsum],[r"$Q_{sum, down}$",'Counts'],[50])
        pau.add_sub_plot(fig,1,3,'2d',[uqsum,dqsum],[r"$Q_{sum, up}$",r"$Q_{sum, down}$"],[50, 50], [False,False,False],[0,20e3],[0,20e3],True)
        pau.add_sub_plot(fig,2,1,'1d',[uxg],[r"$x_{g, up}$",'Counts'],[50])
        pau.add_sub_plot(fig,2,2,'1d',[dxg],[r"$x_{g, down}$",'Counts'],[50])
        pau.add_sub_plot(fig,2,3,'2d',[uxg,dxg],[r"$x_{g, up}$",r"$x_{g, down}$"],[50, 50], [False,False,False],[-20,20],[-20,20],True)

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
                    thickness=20,
                    thicknessmode="pixels",
                    x=xa[1] + 0.01,
                    y=(ya[0] + ya[1]) / 2,
                    len=ya[1] - ya[0] 
                ))

        fig.update_layout( height=800, width=1600, showlegend=False,title_text=f"{input_finename}")
        fig.show()
        # saveutil.save_plotly(fig, base_dir=savebase)

    if 0:
        saveutil.generate_gif(
            input_dir=f"{savebase}/20250907",   # フレーム画像がある日付フォルダ
            output_dir=f"{savebase}/gifs",      # 出力の基底フォルダ
            duration_s=0.2,                     # 1フレーム 0.1 秒
            pattern="*.png",                    # 入力拡張子
            sort_by="ctime",                    # 作成時刻でソート
        )

if __name__ == '__main__':
    main()
