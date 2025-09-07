"""!
@file pyspark_event_builder_version2.py
@version 1
@author Fumitaka ENDO
@date 2025-09-07T19:32:50+09:00
@brief event building from parquet
"""
import click
import pathlib
import sys
import toml
import os
import pandas as pd
from pyspark.sql import functions as F, types as T
from pyspark.sql import SparkSession

this_file_path = pathlib.Path(__file__).parent
sys.path.append(str(this_file_path.parent.parent.parent / "src"))

def common_options(func):
    @click.option('--save', is_flag=True, help='output file generation flag')

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def add_event_id_anchor(df, time_col: str = "t0_ms", threshold_ms: float = 50.0, id_col: str = "event_id"):

    df1 = df.withColumn("__rid", F.monotonically_increasing_id())

    df_nonnull = (
        df1
        .where(F.col(time_col).isNotNull())
        .select("__rid", F.col(time_col).cast("double").alias(time_col))
        .orderBy(F.col(time_col).asc(), F.col("__rid").asc())
        .coalesce(1)
    )

    schema_ids = T.StructType([
        T.StructField("__rid", T.LongType(), False),
        T.StructField(id_col, T.LongType(), False),
    ])

    def _assign_event_id(pdf_iter):
        for pdf in pdf_iter:
            if pdf.empty:
                yield pd.DataFrame({"__rid": [], id_col: []})
                continue

            pdf = pdf.sort_values([time_col, "__rid"]).reset_index(drop=True)
            ev = []
            gid = 1
            anchor = pdf.loc[0, time_col]
            ev.append(gid)
            for v in pdf[time_col].iloc[1:]:
                if abs(v - anchor) <= threshold_ms:
                    ev.append(gid)
                else:
                    gid += 1
                    anchor = v 
                    ev.append(gid)
            out = pd.DataFrame({"__rid": pdf["__rid"], id_col: ev})
            yield out

    df_ids = df_nonnull.mapInPandas(_assign_event_id, schema=schema_ids)
    df_out = (df1.join(df_ids, on="__rid", how="left").drop("__rid"))

    return df_out

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@common_options
def main(save):

    toml_file_path = this_file_path  / "../../../parameters.toml"

    with open(toml_file_path, "r") as f:
        config = toml.load(f)

    fileinfo = config["fileinfo"]
    base_path = fileinfo["base_output_path"]  + "/" + fileinfo["input_file_name"] 
    input  = base_path + "_pulse.parquet"
    output = base_path + "_phys.parquet"
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
    # df.show(10)

    pulse_sum = F.aggregate(
        F.col("pulse"),
        F.lit(0.0),
        lambda acc, x: acc + F.coalesce(x.cast("double"), F.lit(0.0))
    ).alias("charge")

    df = df.withColumnRenamed("t0_ms", "timestamp_ms")
    df = df.withColumnRenamed("pulse_index", "indices")
    df = df.withColumnRenamed("pulse_timestamp", "times")
    

    df_aug = (df
        .select("chip", "channel", "timestamp_ms", "pulse", "indices", "times")
        .withColumn("samidare_id", F.col("chip") * F.lit(32) + F.col("channel"))
        .withColumn("maxsample", F.array_max("pulse").cast("long")) 
        .withColumn("charge", pulse_sum)  
    )

    df_ev = add_event_id_anchor(df_aug, time_col="timestamp_ms", threshold_ms=100.0, id_col="event_id")
    # df_ev.show(10)

    if save:
        (df_ev
            .select("chip", "channel", "samidare_id", "event_id", "timestamp_ms", "pulse", "indices", "times", "maxsample", "charge")
            .write
            .mode("overwrite")
            .option("compression", "zstd")
            .parquet(output))

if __name__ == '__main__':
    main()

