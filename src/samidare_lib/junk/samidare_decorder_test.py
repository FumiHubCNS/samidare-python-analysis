"""
@file samidare_decorder_test.py
@version 2
@author Fumitaka ENDO
@date 2025-07-24T10:29:43+09:00
@brief Sample decoder for binary waveform data
"""
import struct
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HEADER_MARKER = 0xafaf
FOOTER_MARKER = 0xfafa
HEADER_SIZE = 4
FOOTER_SIZE = 2
DATA_SIZE = 40
NUM_CHANNELS = 32
MAX_NUM_SAMPLES = 128

OUTPUTPATH = "/Users/fendo/Work/Data/root/sampa-minitpc-test/output"
BASE = "/Users/fendo/Work/Data/root/sampa-minitpc-test/"
FILE = "test_TPC_001.bin"
# FILE = "test_TPC_1kHz_001.bin"
DATA = BASE+FILE
NAME = FILE.removesuffix(".bin")

def read_word_le(b):
    """Read 2-byte little-endian word"""
    return struct.unpack("<H", b)[0]


def extract_10bit_samples(data_bytes):
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
        samples.append(val)

    return samples


def parse_binary_file(filename):
    """Parse the binary file and dump waveform samples"""
    ch = [[0 for _ in range(MAX_NUM_SAMPLES)] for _ in range(NUM_CHANNELS)]
    current_sample = 0
    current_chip = 0
    events = 0

    with open(filename, "rb") as f:
        data = f.read()

    all_samples=[]
    blockData = []
    loopj = 0
    previous_sample = -1

    offset = 0
    event_id = 0
    raw_parquet_buffer = []

    while offset + HEADER_SIZE + DATA_SIZE + FOOTER_SIZE <= len(data):
        header = read_word_le(data[offset:offset + 2])
        if header != HEADER_MARKER:
            offset += 1
            continue

        header_data = read_word_le(data[offset + 2:offset + 4])
        sample_number = header_data & 0xFF
        chip_number = (header_data >> 8) & 0xFF

        footer = read_word_le(data[offset + HEADER_SIZE + DATA_SIZE:offset + HEADER_SIZE + DATA_SIZE + FOOTER_SIZE])
        if footer != FOOTER_MARKER:
            offset += 1
            continue

        data_block = data[offset + HEADER_SIZE:offset + HEADER_SIZE + DATA_SIZE]
        values = extract_10bit_samples(data_block)

        if previous_sample != sample_number:
            if len(blockData) != 0:
                transposed = [list(row) for row in zip(*blockData)]
                all_samples.append(transposed)

                for k in range(len(transposed)):
                    raw_parquet_buffer.append([sample_number, k, transposed[k]])

                if 1:
                    if len(raw_parquet_buffer) >= 3000:
                        df = pd.DataFrame(raw_parquet_buffer, columns=["chip", "channel", "samples"])
                        df.to_parquet(f"{OUTPUTPATH}/{NAME}_{event_id:05d}.parquet", index=False)
                        print(f"write parquet to {OUTPUTPATH}/{NAME}_{event_id:05d}.parquet")
                        raw_parquet_buffer = [] 
                        event_id += 1
                    
            blockData = []
        
        blockData.append( values )

        previous_sample = sample_number
        previous_chip = chip_number
        offset += HEADER_SIZE + DATA_SIZE + FOOTER_SIZE
        
        loopj = loopj + 1

        if loopj == 2202279:
            break

    if len(raw_parquet_buffer) > 0:
        df = pd.DataFrame(raw_parquet_buffer, columns=["chip", "channel", "samples"])
        df.to_parquet(f"{OUTPUTPATH}/{NAME}_{event_id:05d}.parquet", index=False)
        event_id += 1

    print(f"[Done] Total events: {loopj}")


if __name__ == "__main__":
    parse_binary_file(DATA)
