"""!
@file samidare_decorder_version2.py
@version 1
@author FumiHubCNS
@date 2025-08-20T00:08:14+09:00
@brief template text
"""
import struct
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml

this_file_path = pathlib.Path(__file__).parent

def pretty_print_2byte_blocks(data, blocks_per_line=8):
    assert len(data) % 2 == 0, "データの長さは偶数である必要があります"
    words = [f"{data[i+1]:02x}{data[i]:02x}" for i in range(0, len(data), 2)]
    
    for i in range(0, len(words), blocks_per_line):
        line = " ".join(words[i:i+blocks_per_line])
        print(line)

def read_word_le(b):
    """2バイトリトルエンディアン整数"""
    return struct.unpack("<H", b)[0]

def read_word_be(b):
    """2バイトビックエンディアン整数"""
    return struct.unpack(">H", b)[0]

def extract_timestamp_bytes(timestamp_bytes,debug):
    """Extract 48-bit timestamp from 6 bytes (t3, t2, t1)"""
    # data = list(timestamp_bytes)
    # t3 = (data[3] << 8) | data[2]  # Little-endian
    # t2 = (data[7] << 8) | data[6]
    # t1 = (data[11] << 8) | data[10]
    t3 = read_word_be(timestamp_bytes[2:4])
    t2 = read_word_be(timestamp_bytes[6:8])
    t1 = read_word_be(timestamp_bytes[10:12])
    timestamp = (t3 << 32) | (t2 << 16) | t1
    if debug:
        print(f"t3={hex(t3)}, t2={hex(t2)}, t1={hex(t1)} => timestamp={timestamp}, ({hex(timestamp)})")
    return timestamp

def extract_10bit_samples(data_bytes, NUM_CHANNELS,debug=False):
    """Extract 32 10-bit values from 40 bytes"""
    data = list(data_bytes)
    samples = []

    for i in range(NUM_CHANNELS):
        byte_index = (i * 10) // 8
        offset = i % 4
        if offset == 0:
            val = (data[byte_index] << 2) | ((data[byte_index + 1] & 0xC0) >> 6)
        elif offset == 1:
            val = ((data[byte_index] & 0x3F) << 4) | ((data[byte_index + 1] & 0xF0) >> 4)
        elif offset == 2:
            val = ((data[byte_index] & 0x0F) << 6) | ((data[byte_index + 1] & 0xFC) >> 2)
        elif offset == 3:
            val = ((data[byte_index] & 0x03) << 8) | data[byte_index + 1]

        if debug:
            print(f"loop: {i}, {hex(val)} -> {val}, offset: {offset}, original byte val: {hex(data[byte_index])} {hex(data[byte_index+1])}")
        
        samples.append(val)
    
    return samples
    # reversed_list = samples[::-1]
    # return reversed_list


def parse_binary_file_with_timestamp(filename, HEADER_MARKER, FOOTER_MARKER, HEADER_SIZE, \
    TN_SIZE, FOOTER_SIZE, TIMESTAMP_SIZE, DATA_SIZE, NUM_CHANNELS, MAX_NUM_SAMPLES, OUTPUTPATH, NAME, \
        maxloop=-1, debug=False):
    """Parse binary file and dump waveform samples with timestamp"""

    if debug:
        print(f"this method is excuted by debug mode. decoded data is not saved.")

    output_paths = []

    with open(filename, "rb") as f:
        data = f.read()

    # all_samples=[]
    blockData = []
    loopj = 0
    previous_sample = -1

    offset = 0
    event_id = 0
    raw_parquet_buffer = []

    BLOCK_SIZE = HEADER_SIZE + TN_SIZE + TIMESTAMP_SIZE + DATA_SIZE + FOOTER_SIZE 

    while offset + BLOCK_SIZE <= len(data):
        if read_word_le(data[offset:offset + 2]) != HEADER_MARKER:
            offset += 1
            continue

        header_data = read_word_le(data[offset + 2:offset + 4])
        sample_number = header_data & 0xFF
        chip_number = (header_data >> 8) & 0xFF

        if debug:
            print(f"chip: {sample_number}, sample number: {chip_number}")

        timestamp_bytes = data[offset + HEADER_SIZE: offset + HEADER_SIZE + TN_SIZE + TIMESTAMP_SIZE]
        timestamp = extract_timestamp_bytes(timestamp_bytes,debug)

        footer = read_word_le(data[offset + BLOCK_SIZE - FOOTER_SIZE: offset + BLOCK_SIZE])
        if footer != FOOTER_MARKER:
            offset += 1
            continue

        data_block = data[offset + HEADER_SIZE + TN_SIZE + TIMESTAMP_SIZE: offset + HEADER_SIZE + TN_SIZE + TIMESTAMP_SIZE + DATA_SIZE]
        values = extract_10bit_samples(data_block, NUM_CHANNELS,debug)

        if previous_sample != sample_number:
            if len(blockData) != 0:
                transposed = [list(row) for row in zip(*blockData)]

                for k in range(len(transposed)):
                    raw_parquet_buffer.append([
                        sample_number,
                        k,
                        transposed[k],
                        timestamp
                    ])

                if not debug:
                    if len(raw_parquet_buffer) >= 500000:
                        df = pd.DataFrame(raw_parquet_buffer, columns=["chip", "channel", "samples", "timestamp"])
                        df.to_parquet(f"{OUTPUTPATH}/{NAME}_{event_id:05d}.parquet", index=False)
                        print(f"write parquet to {OUTPUTPATH}/{NAME}_{event_id:05d}.parquet")
                        raw_parquet_buffer = [] 
                        event_id += 1

            blockData = []
        
        blockData.append(values)

        previous_sample = sample_number
        # previous_chip = chip_number
        offset += BLOCK_SIZE
        loopj += 1

        if 1:
            if loopj == maxloop:
                print("event numuber reached {maxloop}. loop process wad stoped")
                break


    if not debug:
        if len(raw_parquet_buffer) > 0:
            df = pd.DataFrame(raw_parquet_buffer, columns=["chip", "channel", "samples", "timestamp"])
            df.to_parquet(f"{OUTPUTPATH}/{NAME}_{event_id:05d}.parquet", index=False)
            print(f"write parquet to {OUTPUTPATH}/{NAME}_{event_id:05d}.parquet")
            output_paths.append(f"{OUTPUTPATH}/{NAME}_{event_id:05d}.parquet")
            event_id += 1

    print(f"[Done] Total events: {loopj}")

    return output_paths


if __name__ == "__main__":

    toml_file_path = this_file_path  / "../../../parameters.toml"

    with open(toml_file_path, "r") as f:
        config = toml.load(f)

    format = config["format"]

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


    base_input_path  = "/Users/fendo/Work/Data/root/sampa-minitpc-test/data2025"
    input_file_dir   = "0919" 
    input_file_name  = "20250919_test_pos_4mVfC_300ns_64sample_16presample_395thre_001"
    base_output_path = "/Users/fendo/Work/Data/root/sampa-minitpc-test/output"

    OUTPUTPATH = "/Users/fendo/Work/Data/root/sampa-minitpc-test/output"
    BASE = base_input_path + "/" + input_file_dir + "/"
    FILE = input_file_name + ".bin"



    DATA = BASE+FILE
    NAME = FILE.removesuffix(".bin")

    parse_binary_file_with_timestamp( DATA, HEADER_MARKER, FOOTER_MARKER, HEADER_SIZE, \
        TN_SIZE, FOOTER_SIZE, TIMESTAMP_SIZE, DATA_SIZE, NUM_CHANNELS, MAX_NUM_SAMPLES, OUTPUTPATH, NAME, \
            10, True)