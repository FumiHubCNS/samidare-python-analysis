"""!
@file samidare_decorder_test_64packet.py
@version 1
@author Fumitaka ENDO
@date 2025-07-26T07:45:30+09:00
@brief sample decorder
"""
import argparse
import pathlib

this_file_path = pathlib.Path(__file__).parent
import struct

HEADER_MARKER = 0xafaf
FOOTER_MARKER = 0xfafa
HEADER_SIZE = 4
FOOTER_SIZE = 2
DATA_SIZE = 40
NUM_CHANNELS = 32
MAX_NUM_SAMPLES = 64  # ← 64サンプル分が1イベント

DATA_FILE = "/Users/fendo/Work/Data/root/sampa-minitpc-test/test_TPC_001.bin"

def read_word_le(b):
    return struct.unpack("<H", b)[0]

def extract_10bit_samples(data_bytes):
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

def parse_packets(filename):
    with open(filename, "rb") as f:
        data = f.read()

    offset = 0
    ch = [[0] * MAX_NUM_SAMPLES for _ in range(NUM_CHANNELS)]
    filled_samples = set()
    events = 0

    while offset + HEADER_SIZE + DATA_SIZE + FOOTER_SIZE <= len(data):
        header = read_word_le(data[offset:offset+2])
        if header != HEADER_MARKER:
            offset += 1
            continue

        header_data = read_word_le(data[offset+2:offset+4])
        sample_number = header_data & 0xFF
        chip_number = (header_data >> 8) & 0xFF

        footer = read_word_le(data[offset + HEADER_SIZE + DATA_SIZE : offset + HEADER_SIZE + DATA_SIZE + FOOTER_SIZE])
        if footer != FOOTER_MARKER:
            offset += 1
            continue

        data_block = data[offset + HEADER_SIZE : offset + HEADER_SIZE + DATA_SIZE]
        values = extract_10bit_samples(data_block)

        if sample_number >= MAX_NUM_SAMPLES:
            print(f"[Warning] Ignoring sample {sample_number} out of expected range")
            offset += HEADER_SIZE + DATA_SIZE + FOOTER_SIZE
            continue

        for ch_index in range(NUM_CHANNELS):
            ch[ch_index][sample_number] = values[ch_index]
        
        filled_samples.add(sample_number)

        # イベント完成: 全64サンプルが揃ったら出力
        if len(filled_samples) == MAX_NUM_SAMPLES:
            print(f"\n[Event {events}] Chip {chip_number}")
            for ch_index in range(NUM_CHANNELS):
                waveform = ", ".join(f"{v:4}" for v in ch[ch_index])
                print(f"ch[{ch_index:02d}] = [{waveform}]")
            events += 1
            filled_samples.clear()
            ch = [[0] * MAX_NUM_SAMPLES for _ in range(NUM_CHANNELS)]

        offset += HEADER_SIZE + DATA_SIZE + FOOTER_SIZE

    print(f"\n[Done] Total events decoded: {events}")

if __name__ == "__main__":
    parse_packets(DATA_FILE)
