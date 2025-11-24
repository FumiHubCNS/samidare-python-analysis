"""!
@file pyspark_event_build.py
@version 1
@author FumiHubCNS
@date 2025-08-20T01:32:36+09:00
@brief template text
"""
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, struct, udf, explode
from pyspark.sql.types import *
import pandas as pd
import plotly.express as px
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, ArrayType
from pyspark.sql import SparkSession
from pyspark.sql.functions import size
from pyspark.sql.functions import size, array_max, array_min
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# 入出力パス
BASE = "/Users/fendo/Work/Data/root/sampa-minitpc-test/output/"
FILE = "20250919_test_pos_4mVfC_300ns_64sample_16presample_395thre_001_00000"

INPUT_PARQUET_PATH = BASE+FILE+".parquet"
OUTPUT_PARQUET_PATH = BASE+FILE+"_event.parquet"

DELTA_T = 100000  # イベント内の最大タイム差

def build_spark():
    # Sparkセッション
    spark = SparkSession.builder.appName("SparkEventBuilder").getOrCreate()

    # 読み込み
    df = spark.read.parquet(INPUT_PARQUET_PATH)

    # ウィンドウ定義（timestamp順）
    w = Window.orderBy("timestamp")

    # 前の行の timestamp を取得
    df = df.withColumn("prev_ts", F.lag("timestamp", 1).over(w))

    # ジャンプがあればフラグ1、それ以外は0
    df = df.withColumn("jump_flag", F.when(
        (F.col("prev_ts").isNull()) | (F.col("timestamp") - F.col("prev_ts") > DELTA_T), 1
    ).otherwise(0))

    # ジャンプフラグを累積和して event_id に
    df = df.withColumn("event_id", F.sum("jump_flag").over(w.rowsBetween(Window.unboundedPreceding, 0)))

    # イベント単位で集約
    rebuilt_df = df.groupBy("event_id").agg(
        F.collect_list("chip").alias("chips"),
        F.collect_list("channel").alias("channels"),
        F.collect_list("timestamp").alias("timestamps"),
        F.collect_list("samples").alias("samples")
    )

    # 書き出し
    rebuilt_df.write.mode("overwrite").parquet(OUTPUT_PARQUET_PATH)

    print("[DONE] Sparkのみでイベント構築＆保存完了")


def build_local():
    # SparkSession 作成
    spark = SparkSession.builder.appName("EventRebuilder").getOrCreate()

    # 元データ読み込み
    df = spark.read.parquet(INPUT_PARQUET_PATH)

    # ソート（PySparkではWindow関数を使わず collect → group）
    df_sorted = df.orderBy("timestamp")

    # pandasに一旦変換（ローカルメモリに乗る程度の量である前提）
    pdf = df_sorted.toPandas()

    # イベント構築
    events = []
    current_event = []
    current_event_id = 0
    prev_ts = None

    for idx, row in pdf.iterrows():
        ts = row["timestamp"]
        if prev_ts is None or ts - prev_ts <= DELTA_T:
            current_event.append(row)
        else:
            events.append((current_event_id, current_event))
            current_event = [row]
            current_event_id += 1
        prev_ts = ts

    # 最後のイベント追加
    if current_event:
        events.append((current_event_id, current_event))

    # イベントごとに集約データ作成
    rebuilt_data = []
    for eid, event_rows in events:
        chips = []
        chans = []
        times = []
        samps = []

        for r in event_rows:
            chips.append(r["chip"])
            chans.append(r["channel"])
            times.append(r["timestamp"])
            samps.append(r["samples"])

        rebuilt_data.append({
            "event_id": eid,
            "chips": chips,
            "channels": chans,
            "timestamps": times,
            "samples": samps
        })

    # 新しいDataFrameをSparkに戻す
    rebuilt_df = spark.createDataFrame(pd.DataFrame(rebuilt_data))
    rebuilt_df.write.mode("overwrite").parquet(OUTPUT_PARQUET_PATH)

    print("[DONE] イベント構築と保存完了")

def check_output(panqet_path=OUTPUT_PARQUET_PATH):

    spark = SparkSession.builder.appName("MySparkApp").getOrCreate()
    data = spark.read.parquet(panqet_path)
    data.show(10, truncate=False)


def check_timestamp(panqet_path=INPUT_PARQUET_PATH):
    # Sparkセッション
    spark = SparkSession.builder.appName("MySparkApp").getOrCreate()

    # Parquet 読み込み
    df = spark.read.parquet(panqet_path)
    df_sorted = df.orderBy("timestamp")

    # タイムスタンプだけ抽出して Pandas に変換
    timestamp_df = df_sorted.select("timestamp").limit(50000).toPandas()

    # 行番号を追加
    timestamp_df["index"] = timestamp_df.index

    # Plotly でプロット
    fig = px.line(timestamp_df, x="index", y="timestamp", title="Timestamp over Row Index")
    fig.show()

def check_event_legth(panqet_path=OUTPUT_PARQUET_PATH):
    # SparkSession 作成
    spark = SparkSession.builder.appName("HistogramEventLengths").getOrCreate()

    # 再構築済みイベントの Parquet 読み込み
    df = spark.read.parquet(panqet_path)

    # timestamps のリスト長を取得する新しいカラムを追加
    df_with_length = df.withColumn("timestamp_length", size("timestamps"))
    # Pandasに変換（軽量な前提）
    lengths_pdf = df_with_length.select("timestamp_length").toPandas()

    fig = px.histogram(lengths_pdf, x="timestamp_length", nbins=30,
                    title="Event Size Distribution (Number of timestamps per event)",
                    labels={"timestamp_length": "Timestamps per Event"})
    fig.show()

def check_event_legth_vs_dt(panqet_path=OUTPUT_PARQUET_PATH):
    # Sparkセッション作成
    spark = SparkSession.builder.appName("EventHeatmap").getOrCreate()

    # Parquet読み込み
    df = spark.read.parquet(panqet_path)

    # 要素数と最大−最小の差分を求める
    df_metrics = df.select(
        size("timestamps").alias("num_timestamps"),
        (array_max("timestamps") - array_min("timestamps")).alias("ts_range")
    )

    # pandasへ変換
    pdf = df_metrics.toPandas()

    # # Plotlyで2Dヒートマップ
    # fig = px.density_heatmap(
    #     pdf,
    #     x="num_timestamps",
    #     y="ts_range",
    #     nbinsx=200,
    #     nbinsy=200,
    #     color_continuous_scale="Blues",
    #     title="2D Frequency Heatmap: Timestamps per Event vs. Time Range"
    # )
    # fig.update_layout(
    #     xaxis_title="Number of Timestamps",
    #     yaxis_title="Timestamp Range (max - min)"
    # )
    # fig.show()
    
    pdf["event_id"] = pdf.index

    # サブプロット作成
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "2D Heatmap: Timestamps Count vs Timestamp Range ( Max - Min )",
            "1D Histogram: Event hit length",
            "1D Histogram: Timestamp Range ( Max - Min )",
            "Scatter: Event ID vs Timestamp Range ( Max - Min )"
        ),
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )

    # 1️⃣ 2D ヒートマップ（go.Histogram2dで正確に制御）
    heatmap = go.Histogram2d(
        x=pdf["num_timestamps"],
        y=pdf["ts_range"],
        nbinsx=200,
        nbinsy=200,
        colorscale="Blues",
        colorbar=dict(title="Count")
    )
    fig.add_trace(heatmap, row=1, col=1)
    fig.update_xaxes(title_text="Number of Timestamps", range=[0, 300], row=1, col=1)
    fig.update_yaxes(title_text="Timestamp Range ( Max - Min )", range=[0, 150000], row=1, col=1)

    # 2️⃣ ヒストグラム（x軸：num_timestamps）
    hist_x = go.Histogram(
        x=pdf["num_timestamps"],
        nbinsx=100,
        marker=dict(color="skyblue")
    )
    fig.add_trace(hist_x, row=1, col=2)
    fig.update_xaxes(title_text="Number of Timestamps", range=[0, 300], row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)

    # 3️⃣ ヒストグラム（x軸：ts_range）
    hist_y = go.Histogram(
        x=pdf["ts_range"],
        nbinsx=100,
        marker=dict(color="salmon")
    )
    fig.add_trace(hist_y, row=2, col=1)
    fig.update_xaxes(title_text="Timestamp Range ( Max - Min )", range=[0, 150000], row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    # 4️⃣ 散布図（event_id vs ts_range）
    scatter = go.Scattergl(
        x=pdf["event_id"],
        y=pdf["ts_range"],
        mode="markers",
        marker=dict(size=2, color="black")
    )
    fig.add_trace(scatter, row=2, col=2)
    fig.update_xaxes(title_text="Event ID", range=[0, len(pdf)], row=2, col=2)
    fig.update_yaxes(title_text="Timestamp Range ( Max - Min )", range=[0, 150000], row=2, col=2)

    # 全体のレイアウト
    fig.update_layout(
        height=900,
        width=1200,
        title_text="Event Timestamp Statistics Overview",
        showlegend=False
    )

    fig.show()

if __name__ == "__main__":
    # build_spark()
    # check_output()
    # check_timestamp()
    # check_event_legth()
    # check_event_legth_vs_dt()

