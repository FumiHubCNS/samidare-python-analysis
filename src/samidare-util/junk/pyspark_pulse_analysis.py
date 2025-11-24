"""!
@file pyspark_pulse_analysis.py
@version 1
@author FumiHubCNS
@date 2025-08-20T13:23:12+09:00
@brief template text
"""
import click
import pathlib
import datetime

this_file_path = pathlib.Path(__file__).parent

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, StringType, StructType, StructField, LongType
from pyspark.sql.types import ArrayType, IntegerType
from pyspark.sql import SparkSession
import pandas as pd

def extract_valid_segments(event_id, chips, chans, timestamps, samples,
                           threshold_up=100, threshold_down=100,
                           pre_samples=12, post_samples=12,
                           baseline_samples=10):
    filtered_chips = []
    filtered_chans = []
    filtered_timestamps = []
    filtered_samples = []

    for i in range(len(samples)):
        waveform = samples[i]
        if not waveform or len(waveform) < 20:
            continue

        front = waveform[:11]
        back = waveform[-11:]
        combined = front + back
        if not combined:
            continue
        baseline_1 = sum(combined) / len(combined)
        corrected_waveform = [v - baseline_1 for v in waveform]

        rise_index = next((j for j, v in enumerate(corrected_waveform) if v >= threshold_up), None)
        if rise_index is None:
            continue

        fall_index = next(
            (j for j in range(rise_index, len(corrected_waveform)) if corrected_waveform[j] < threshold_down),
            len(corrected_waveform) - 1
        )

        start = max(0, rise_index - pre_samples)
        end = min(len(corrected_waveform), fall_index + post_samples)
        segment = corrected_waveform[start:end]

        if len(segment) < baseline_samples:
            continue

        baseline_2 = sum(segment[:baseline_samples]) / baseline_samples
        final_segment = [v - baseline_2 for v in segment]

        filtered_chips.append(chips[i])
        filtered_chans.append(chans[i])
        filtered_timestamps.append(timestamps[i])
        filtered_samples.append(final_segment)

    return (event_id, filtered_chips, filtered_chans, filtered_timestamps, filtered_samples)

BASE = "/Users/fendo/Work/Data/root/sampa-minitpc-test/output/"
FILE = "20250919_test_pos_4mVfC_300ns_64sample_16presample_395thre_001_00000"

INPUT_PARQUET_PATH = BASE+FILE+"_event.parquet"
OUTPUT_PARQUET_PATH = BASE + FILE + "_filtered.parquet"

# spark = SparkSession.builder.appName("WaveformProcessing").getOrCreate()
# df = spark.read.parquet(INPUT_PARQUET_PATH)

# df_pd = df.select("event_id","chips", "channels", "timestamps", "samples").limit(1).toPandas()
# df_sample = df.limit(100).cache()
# df_pd = df_sample.toPandas()

# results = df_pd.apply(
#     lambda row: extract_valid_segments(
#         row['event_id'],
#         row['chips'],
#         row['channels'],
#         row['timestamps'],
#         row['samples']
#     ),
#     axis=1
# )

# results.write.mode("overwrite").parquet(OUTPUT_PARQUET_PATH)

# print("[DONE] filtered waveform を Parquet に保存しました")

# result_df = pd.DataFrame(results.tolist(), columns=[
#     "filtered_event_id", "filtered_chips", "filtered_channels", "filtered_timestamps", "filtered_samples"
# ])

# print(result_df.head(3))

import pyarrow.parquet as pq
import os

# ディレクトリ内の1つのファイルを指定
event_parquet_dir = INPUT_PARQUET_PATH
parquet_files = [f for f in os.listdir(event_parquet_dir) if f.endswith(".parquet") or f.endswith(".snappy.parquet")]

# 1ファイル選んで読み込む
file_path = os.path.join(event_parquet_dir, parquet_files[0])
parquet_file = pq.ParquetFile(file_path)

# バッチ単位で読み込み（batch_size=1にすれば1行ずつ）
i = 0

for batch in parquet_file.iter_batches(batch_size=10):
    table = batch.to_pydict()

    event_id = table["event_id"]
    chips = table["chips"]
    chans = table["channels"]
    timestamps = table["timestamps"]
    samples = table["samples"]

    print(f"Event ID: {i}, {event_id}")
    # print(f"Timestamps: {timestamps[:5]}")
    # print(f"Sample count: {len(samples)}")
    #

    print(f"chips: {type(chips)} length : {len(chips[9])}")
    print(f"channels: {type(chans)} length : {len(chans)}")
    print(f"timestamps: {type(timestamps)} length : {len(timestamps)}")
    print(f"chansamplesnels: {type(samples)} length : {len(samples)}")



    # print("chips element:", type(table["chips"][0]) if table["chips"] else "Empty")
    # print("samples element:", type(table["samples"][0]) if table["samples"] else "Empty")
    # print("1 waveform sample element:", type(table["samples"][0][0]) if table["samples"] and table["samples"][0] else "Empty")
    print("-----")

    # result = extract_valid_segments(i, chips, chans, timestamps, samples)

    if i >3:
        break

    i = i + 1 

