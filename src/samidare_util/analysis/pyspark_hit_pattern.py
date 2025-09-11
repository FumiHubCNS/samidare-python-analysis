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
import matplotlib.pyplot as plt
import numpy as np
import os 
import plotly.graph_objs as go

from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql import functions as F, types as T
from plotly.subplots import make_subplots
from math import isfinite
import numpy as np
from typing import Sequence, Tuple, Dict, Any

import catmlib as cat
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


def check_fit_alg(x,y,q):

    chi2, res = fit_global_odr(x, y, q)
    print("chi2 =", chi2)
    print("angle(deg) =", res["angle_deg"])
    print("line: y = {:.6g} x + {}".format(res["slope"], res["intercept"]))


    # 数値結果を表に
    df = pd.DataFrame([{
        "chi2": chi2,
        "chi2_reduced": res["chi2_reduced"],
        "slope": res["slope"],
        "intercept": res["intercept"],
        "angle_deg": res["angle_deg"],
        "n_points": res["n_points"],
        "dof": res["dof"],
        "centroid_x": res["centroid"][0],
        "centroid_y": res["centroid"][1],
    }])

    # 可視化（データ点とフィット直線）
    X = np.asarray(x, dtype=float)
    Y = np.asarray(y, dtype=float)
    nx, ny, c = res["normal"]

    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    xm = (xmax - xmin) if xmax > xmin else 1.0
    ym = (ymax - ymin) if ymax > ymin else 1.0
    padx = 0.1 * xm
    pady = 0.1 * ym

    fig = plt.figure(figsize=(6, 4.5))

    # 散布図（重みはサイズに反映）
    sizes = (np.asarray(q) / np.max(q)) * 80 + 20  # 20〜100あたり
    plt.scatter(X, Y, s=sizes, label="data (size ~ weight)")

    # フィット直線の描画
    if np.isfinite(res["slope"]):
        xs = np.linspace(xmin - padx, xmax + padx, 200)
        ys = res["slope"] * xs + (res["intercept"] if res["intercept"] is not None else 0.0)
        plt.plot(xs, ys, label="fit line")
    else:
        # 垂直直線 x = -c/nx
        x0 = -c / nx
        ys = np.linspace(ymin - pady, ymax + pady, 200)
        plt.plot([x0, x0], [ys.min(), ys.max()], label="fit line (vertical)")

    plt.xlabel("x")
    plt.ylabel("y")
    title = f"Global ODR fit  (χ² = {chi2:.4g}"
    if res["chi2_reduced"] is not None:
        title += f",  χ²/ν = {res['chi2_reduced']:.4g}"
    title += f",  angle = {res['angle_deg']:.3g}°)"
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def fit_global_odr(x: Sequence[float],
                   y: Sequence[float],
                   q: Sequence[float],
                   *,
                   return_reduced_chi2: bool = True
                   ) -> Tuple[float, Dict[str, Any]]:
    """
    Weighted Orthogonal Distance Regression (Global Fitting).
    - 入力: x, y, q（同じ長さ）
      q は各点の重み（例: 誘起電荷）。負や0、非有限は除外。
    - 目的関数: χ² = Σ_i q_i * d_i^2
      ここで d_i は点 (x_i, y_i) と直線の「直交距離」。
    - 出力:
        chi2: 上式の χ²
        result(dict):
          - slope: 直線 y = a x + b の a（ほぼ垂直なら np.inf）
          - intercept: b（ほぼ垂直なら None）
          - normal: (nx, ny, c) 単位法線での行列表現 nx*x + ny*y + c = 0
          - angle_rad / angle_deg: 直線方向ベクトルの角度（x軸に対して）
          - centroid: (cx, cy) 加重重心
          - n_points: 使用点数
          - dof: 自由度（= n_points - 2、<0なら 0）
          - chi2_reduced: χ²/DOF（計算可能な場合のみ）
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(q, dtype=float)

    if x.shape != y.shape or x.shape != w.shape:
        raise ValueError("x, y, q の長さは同じである必要があります。")

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        raise ValueError("有効なデータ点がありません（非有限/重み<=0 しかない）。")

    X = x[mask]
    Y = y[mask]
    W = w[mask]
    n_points = X.size
    if n_points < 2:
        raise ValueError("フィットには少なくとも2点が必要です。")

    # --- 加重重心
    Wsum = W.sum()
    cx = (W * X).sum() / Wsum
    cy = (W * Y).sum() / Wsum

    # --- 加重共分散行列（PCAで主軸=直線方向を得る）
    dx = X - cx
    dy = Y - cy
    Sxx = (W * dx * dx).sum() / Wsum
    Syy = (W * dy * dy).sum() / Wsum
    Sxy = (W * dx * dy).sum() / Wsum
    S = np.array([[Sxx, Sxy],
                  [Sxy, Syy]], dtype=float)

    # 固有分解（対象行列）
    vals, vecs = np.linalg.eigh(S)  # 昇順の固有値が返る
    d = vecs[:, np.argmax(vals)]    # 最大固有値に対応する固有ベクトル = 直線方向
    d = d / np.linalg.norm(d)

    # 法線ベクトル（単位）と c： nx*x + ny*y + c = 0
    n = np.array([-d[1], d[0]], dtype=float)  # 方向ベクトルに直交
    n = n / np.linalg.norm(n)
    c = - (n[0] * cx + n[1] * cy)

    # --- χ²（直交距離の重み付き二乗和）を計算
    di = n[0] * X + n[1] * Y + c          # 単位法線なので |di| が直交距離
    chi2 = float((W * di**2).sum())

    # y = a x + b 形式（ほぼ垂直の扱い）
    eps = 1e-12
    if abs(n[1]) < eps:
        slope = float('inf')
        intercept = None
    else:
        slope = - n[0] / n[1]
        intercept = - c / n[1]

    # 角度（直線方向ベクトル d の角度）
    angle_rad = float(np.arctan2(d[1], d[0]))
    angle_deg = float(np.degrees(angle_rad))

    dof = max(n_points - 2, 0)
    chi2_reduced = (chi2 / dof) if (return_reduced_chi2 and dof > 0) else None

    result = {
        "slope": slope,
        "intercept": intercept,
        "normal": (float(n[0]), float(n[1]), float(c)),
        "angle_rad": angle_rad,
        "angle_deg": angle_deg,
        "centroid": (float(cx), float(cy)),
        "n_points": int(n_points),
        "dof": int(dof),
        "chi2_reduced": None if chi2_reduced is None else float(chi2_reduced),
    }
    return chi2, result

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


def calculate_position_charge(
        pad:cat.basepad.TReadoutPadArray = None, 
        vids:list = None, 
        tpcid:list =None,
        charge:list = None,
        time:list = None
        ): 
    ps = []
    qs= []
    
    for idx in vids:    
        id = padinfo.find_index(pad.ids, int(tpcid[int(idx)]))
        pos = pad.centers[id]
        pos[1] = time[idx]
        ps.append(pos)
        qs.append(charge[idx])

    return (ps, qs)


def calculate_weighted_average(vals, wts):
    return sum(v*w for v, w in zip(vals, wts)) / sum(wts)

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

    df_events = (
        df_plot
        .groupBy("event_id")
        .agg(
            F.min("timestamp_ms").alias("event_time_ms"),  # 並び替え用の代表時刻
            F.collect_list(F.struct(
                F.col("timestamp_ms"),
                F.col("tpc_id"),
                F.col("charge"),
                F.col("pulse"),
                F.col("times"),
                F.col("event_id"),
                F.col("samidare_id"),
            )).alias("hits")
        )
        .orderBy("event_id")  # ここで本当にグローバルに並びます
    )

    qsum_arr = []
    tmdiff_arr = []
    tpcid_hist=[]
    samidareid_hist=[]

    uqsum = []
    dqsum = []
    uxg = []
    dxg = []

    fit_hist_a = []
    fit_hist_b = []
    fit_hist_c = []

    figcount = 0

    tpcs = get_pads()

    for ev in df_events.toLocalIterator():

        rows = sorted(ev['hits'], key=lambda h: (h['timestamp_ms']))

        evtid_arr      = [r['event_id']     for r in rows]
        tpcid_arr      = [r['tpc_id']       for r in rows]
        samidareid_arr = [r['samidare_id']  for r in rows]
        charge_arr     = [r['charge']       for r in rows]        # x a = 126 / 19e3 = 0.006631578947
        timestamp_arr  = [r['timestamp_ms'] for r in rows]
        pulse_arr      = [r['pulse']        for r in rows]   
        clock_arr      = [r['times']        for r in rows]

        maxsample_clock_arr = []

        for i in range(len(pulse_arr)):
            maxsample_timestamp = clock_arr[i][int(np.argmax(pulse_arr[i]))] / 340000.
            
            for j in range(len(pulse_arr[i])):
                clock_arr[i][j] = timestamp_arr[i] + maxsample_timestamp

            samidareid_hist.append(samidareid_arr[i])
            tpcid_hist.append(tpcid_arr[i])
            maxsample_clock_arr.append( clock_arr[i][int(np.argmax(pulse_arr[i]))] )


        count1 = sum(1 for v in tpcid_arr if v <  60)
        count2 = sum(1 for v in tpcid_arr if v >= 60)   


        qsum_arr.append(sum(charge_arr))

        if count2 > 3 and count1 > 3 and 1:

            offset = min(maxsample_clock_arr)

            for i in range(len(maxsample_clock_arr)):
                maxsample_clock_arr[i] = (maxsample_clock_arr[i] - offset)
                tmdiff_arr.append(maxsample_clock_arr[i] * 1.e6)
            
            
            # reflist =  timestamp_arr
            reflist = charge_arr
            q_lst = [0] * len(tpcs.ids)

            for i in range(len(reflist)):
                id = padinfo.find_index(tpcs.ids, int(tpcid_arr[i]))
                q_lst[id] = int(reflist[i])
                
            cehck_list = q_lst
            bins, colors = catview.get_color_list(cehck_list, cmap_name="ocean_r", fmt="hex")
            color_array  = catview.get_color_array(cehck_list,bins,colors)

            up_vidx = within_range_indices(tpcid_arr, 0, 59)
            dn_vidx = within_range_indices(tpcid_arr, 60, 119)
            al_vidx = within_range_indices(tpcid_arr, 0, 119)

            ups, uqs = calculate_position_charge(tpcs, up_vidx, tpcid_arr, charge_arr, maxsample_clock_arr)
            dps, dqs = calculate_position_charge(tpcs, dn_vidx, tpcid_arr, charge_arr, maxsample_clock_arr)
            aps, aqs = calculate_position_charge(tpcs, al_vidx, tpcid_arr, charge_arr, maxsample_clock_arr)

            uxs = [p[0] for p in ups]
            dxs = [p[0] for p in dps]
            axs = [p[0] for p in aps]

            uzs = [p[2] for p in ups]
            dzs = [p[2] for p in dps]
            azs = [p[2] for p in aps]

            if ( sum(uqs) < 30e3 ) and ( sum(dqs) > 20e3 ) and (1):

                uchi2, ures = fit_global_odr(uxs, uzs, uqs)
                dchi2, dres = fit_global_odr(dxs, dzs, dqs)
                achi2, ares = fit_global_odr(axs, azs, aqs)

                # check_fit_alg(uzs, uxs, uqs)

                tracks  = []
                # tracks.append(["line",[1/ures["slope"], -ures["intercept"]/ures["slope"]], [40,141], [1,'blue']])
                # tracks.append(["line",[1/dres["slope"], -dres["intercept"]/dres["slope"]], [40,141], [1,'green']])
                tracks.append(["line",[1/ares["slope"], -ares["intercept"]/ares["slope"]], [40,141], [1,'red']])

                fit_hist_a.append(1/ares["slope"])
                fit_hist_b.append(-ares["intercept"]/ares["slope"])
                fit_hist_c.append(ares["chi2_reduced"])

                if 0:
                    tpcs.show_pads(
                        plot_type='map', 
                        color_map=color_array, 
                        xrange=[-20,20],
                        yrange=[38,142],
                        block_flag=False,
                        savepath = None,#f"{savebase}/20250911/{figcount:5d}.png",
                        check_id = True,
                        check_size=3,
                        check_data = tpcs.ids,
                        canvassize = [8,7],
                        tracks=tracks
                    )

            figcount += 1

            # if abs(1/ares["slope"]) < 0.02:

            uqsum.append(sum(uqs) *0.006631578947)
            dqsum.append(sum(dqs) *0.006631578947)

            uxg.append(calculate_weighted_average(uxs, uqs))
            dxg.append(calculate_weighted_average(dxs, dqs))

    if 1:
        fig = make_subplots(rows=2, cols=2, vertical_spacing=0.15, horizontal_spacing=0.1,
        subplot_titles=( "a", "b", "chi2/dof",  "b vs a" ))

        pau.add_sub_plot(fig,1,1,'1d',[fit_hist_a],['a','Counts'],[100])
        pau.add_sub_plot(fig,1,2,'1d',[fit_hist_b],['b','Counts'],[100])
        pau.add_sub_plot(fig,2,1,'1d',[fit_hist_c],['chi2/dof','Counts'],xrange=[0,6000,60])
        pau.add_sub_plot(fig,2,2,'2d',[fit_hist_b,fit_hist_a],['b','a'],[50, 50])

        for trace in fig.data:
            if isinstance(trace, go.Heatmap):
                xaxis = "xaxis" if trace.xaxis == "x" else "xaxis" + trace.xaxis[1:]
                yaxis = "yaxis" if trace.yaxis == "y" else "yaxis" + trace.yaxis[1:]

                xa = fig.layout[xaxis].domain
                ya = fig.layout[yaxis].domain

                trace.update(colorbar=dict(thickness=20, thicknessmode="pixels", x=xa[1] + 0.01, y=(ya[0] + ya[1]) / 2, len=ya[1] - ya[0]))

        fig.update_layout( height=700, width=1400, showlegend=False,title_text=f"{input_finename}")
        fig.show()
        # saveutil.save_plotly(fig, base_dir=savebase)

    if 0:
        fig = make_subplots(rows=2, cols=2, vertical_spacing=0.15, horizontal_spacing=0.1,
        subplot_titles=( "Total Charge", "Max Sample Timiming Difference" ))

        pau.add_sub_plot(fig,1,1,'1d',[qsum_arr],['Total Charge','Counts'],[200])
        pau.add_sub_plot(fig,1,2,'1d',[tmdiff_arr],[r'$\Delta t [ns]$','Counts'],xrange=[0,1e6,20])
        pau.add_sub_plot(fig,2,1,'1d',[samidareid_hist],['SAMIDARE ID','Counts'],xrange=[0,128,1])
        pau.add_sub_plot(fig,2,2,'1d',[tpcid_hist],[f"TPC ID",'Counts'],xrange=[0,128,1])

        fig.update_layout( height=700, width=1400, showlegend=False,title_text=f"{input_finename}")
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

        pau.add_sub_plot(fig,1,1,'1d',[uqsum],[r"$Q_{sum, up}$",'Counts'],[100])
        pau.add_sub_plot(fig,1,2,'1d',[dqsum],[r"$Q_{sum, down}$",'Counts'],[100])
        pau.add_sub_plot(fig,1,3,'2d',[uqsum,dqsum],[r"$Q_{sum, up}$",r"$Q_{sum, down}$"],[50, 50], [False,False,False])#,[0,20e3],[0,20e3],True)
        pau.add_sub_plot(fig,2,1,'1d',[uxg],[r"$x_{g, up}$",'Counts'],xrange=[-20,20,1])
        pau.add_sub_plot(fig,2,2,'1d',[dxg],[r"$x_{g, down}$",'Counts'],xrange=[-20,20,1])
        pau.add_sub_plot(fig,2,3,'2d',[uxg,dxg],[r"$x_{g, up}$",r"$x_{g, down}$"],[50, 50], [False,False,False],[-20,20],[-20,20],True)

        for trace in fig.data:
            if isinstance(trace, go.Heatmap):
                xaxis = "xaxis" if trace.xaxis == "x" else "xaxis" + trace.xaxis[1:]
                yaxis = "yaxis" if trace.yaxis == "y" else "yaxis" + trace.yaxis[1:]

                xa = fig.layout[xaxis].domain
                ya = fig.layout[yaxis].domain

                trace.update(colorbar=dict(thickness=20, thicknessmode="pixels", x=xa[1] + 0.01, y=(ya[0] + ya[1]) / 2, len=ya[1] - ya[0]))

        fig.update_layout( height=800, width=1600, showlegend=False,title_text=f"{input_finename}")
        fig.show()
        # saveutil.save_plotly(fig, base_dir=savebase)

    if 0:
        saveutil.generate_gif(
            input_dir=f"{savebase}/20250911",   
            output_dir=f"{savebase}/gifs",      
            duration_s=0.2,                   
            pattern="*.png",          
            sort_by="ctime",               
        )

if __name__ == '__main__':
    main()
