"""!
@file chk_timestamp.py
@version 1
@author FumiHubCNS
@date 2025-08-19T23:42:18+09:00
@brief template text
"""
import click
import pathlib
import datetime

this_file_path = pathlib.Path(__file__).parent
import struct
import plotly.graph_objects as go

# 定数
HEADER_MARKER = 0xafaf
FOOTER_MARKER = 0xfafa
HEADER_SIZE = 2
HEADER_INFO_SIZE = 2
TIMESTAMP_SIZE = 6
DATA_SIZE = 40
FOOTER_SIZE = 2
BLOCK_SIZE = HEADER_SIZE + HEADER_INFO_SIZE + TIMESTAMP_SIZE + DATA_SIZE + FOOTER_SIZE

def read_word_le(b):
    """2バイトリトルエンディアン整数"""
    return struct.unpack("<H", b)[0]

def read_word_be(b):
    """2バイトビックエンディアン整数"""
    return struct.unpack(">H", b)[0]

def extract_timestamps(filename, max_blocks=1000):
    timestamps = []

    with open(filename, "rb") as f:
        data = f.read()

    offset = 0
    blocks_read = 0

    while offset + BLOCK_SIZE <= len(data):
        if read_word_le(data[offset:offset+2]) != HEADER_MARKER:
            offset += 1
            continue

        t3 = read_word_be(data[offset+6:offset+8])
        t2 = read_word_be(data[offset+10:offset+12])
        t1 = read_word_be(data[offset+14:offset+16])

        timestamp = (t3 << 32) | (t2 << 16) | t1
        timestamps.append(timestamp)

        offset += BLOCK_SIZE
        blocks_read += 1
        if blocks_read == max_blocks:
            break
    
        print(f"t3={hex(t3)}, t2={hex(t2)}, t1={hex(t1)} => timestamp={timestamp}, ({hex(timestamp)})")

    return timestamps

def plot_timestamps_plotly(timestamps):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=timestamps,
        mode='lines+markers',
        name='timestamp',
        marker=dict(size=4),
        line=dict(width=1)
    ))
    fig.update_layout(
        title='Timestamps from Binary File',
        xaxis_title='Event Index',
        yaxis_title='Timestamp (48-bit)',
        hovermode='x unified',
        height=500
    )
    fig.show()

if __name__ == '__main__':
    base_input_path  = "/Users/fendo/Work/Data/root/sampa-minitpc-test/data2025"
    input_file_dir   = "0919" 
    input_file_name  = "20250919_test_pos_4mVfC_300ns_64sample_16presample_395thre_001"
  
    filename =  base_input_path + "/" + input_file_dir + "/" + input_file_name + ".bin"
    timestamps = extract_timestamps(filename,5000)
    # plot_timestamps_plotly(timestamps)


