"""!
@file pyspark_event_builer.py
@version 1
@author FumiHubCNS
@date 2025-08-22T11:12:32+09:00
@brief template text
"""
import click
import pathlib
import datetime
import toml
import sys

this_file_path = pathlib.Path(__file__).parent
sys.path.append(str(this_file_path.parent.parent.parent / "src"))

import samidare_lib.decoder.samidare_decorder_version2 as samidare_decoder
import samidare_lib.decoder.pyspark_pulse_analysis_version2 as pulse_analysis

def common_options(func):
    # @click.option('--name', '-n', default='World', show_default=True, help='Your name')
    @click.option('--date', '-d', type=click.DateTime(formats=["%Y-%m-%d"]), default=lambda: datetime.datetime.now(), show_default=lambda: datetime.datetime.now().strftime("%Y-%m-%d"), help="Date in YYYY-MM-DD format")
    @click.option('--verbose', '-v', is_flag=True, help='verbose flag')
    @click.option('--check', '-c', is_flag=True, help='data check flag')
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@common_options
def main(date, verbose, check):
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
        HEADER_SIZE, TN_SIZE, FOOTER_SIZE, TIMESTAMP_SIZE, DATA_SIZE, NUM_CHANNELS, MAX_NUM_SAMPLES, OUTPUTPATH, NAME)

    ##### event building and pulse finding
    for i in range(len(output_paths)):
        original_path = output_paths[i]
        modified_path = original_path.replace(".parquet", "_pulse.parquet")

    INPUT_PARQUET_PATH = original_path
    OUTPUT_PARQUET_PATH = modified_path    
    MAX_DELTA = analysinfo["time_window"] 

    pulse_analysis.build_event_and_find_pulse(INPUT_PARQUET_PATH, OUTPUT_PARQUET_PATH, MAX_DELTA, check)

if __name__ == '__main__':
    main()
