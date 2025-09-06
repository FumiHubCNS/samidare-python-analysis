"""!
@file binary_dumper_version3.py
@version 1
@author Fumitaka ENDO
@date 2025-09-04T22:44:03+09:00
@brief Dump binary file
"""
import click
import pathlib
import datetime
import sys
import toml
from typing import Iterable
from typing import List, Optional
from typing import Dict, List, Tuple, Optional
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from pyspark.sql import SparkSession, functions as F, types as T
import samidare_util.decoder.pyspark_pulse_analysis_version2 as pau
from itertools import islice
import matplotlib.pyplot as plt
from statistics import mode
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from typing import Iterable, Union, List, Tuple

this_file_path = pathlib.Path(__file__).parent
sys.path.append(str(this_file_path.parent.parent.parent / "src"))

Pair = Tuple[int, int]  # (pos, byte)
Color = Tuple[float, float, float, float]  # RGBA (0..1)


class PulseParquetAppender:                    
    # sample_build_data = {
    #     "chip" : transposed_sample_block[3][0],
    #     "timestamp": transposed_sample_block[2],
    #     "sample_index":  transposed_sample_block[4],
    #     "sample_value" : fadc_samples
    #  }
    def __init__(self, path: str, batch_rows: int = 8192):
        self.schema = pa.schema([            
            ("chip",          pa.int64()),              
            ("timestamp", pa.list_(pa.int64())),
            ("sample_index", pa.list_(pa.int64())),            
            ("samples_value", pa.list_(pa.list_(pa.int64()))) 
        ])
        self.writer = pq.ParquetWriter(path, self.schema, compression="zstd")
        self.batch_rows = batch_rows
        self._buf = {name: [] for name in self.schema.names}
        self._n = 0

    def _flush(self):
        if self._n == 0:
            return
        arrays = []
        for name, typ in zip(self.schema.names, self.schema.types):
            vals = self._buf[name]
            arrays.append(pa.array(vals, type=typ))
        table = pa.Table.from_arrays(arrays, names=self.schema.names)
        # 行をまとめて 1 つの RowGroup として書く
        self.writer.write_table(table, row_group_size=self._n)
        # バッファをクリア
        self._buf = {name: [] for name in self.schema.names}
        self._n = 0

    def append(self, row: dict):
        for name in self.schema.names:
            self._buf[name].append(row.get(name))
        self._n += 1
        if self._n >= self.batch_rows:
            self._flush()

    def close(self):
        self._flush()
        self.writer.close()
class SAMPADataParquetAppender:
    def __init__(self, path: str, batch_rows: int = 8192):
        self.schema = pa.schema([
            ("data_block",    pa.int64()),
            ("error_level",   pa.int64()),
            ("timestamp",     pa.int64()),
            ("chip",          pa.int64()),
            ("sample_index",  pa.int64()),
            ("samples_value", pa.list_(pa.int64())),
        ])
        self.writer = pq.ParquetWriter(path, self.schema, compression="zstd")
        self.batch_rows = batch_rows
        self._buf = {name: [] for name in self.schema.names}
        self._n = 0

    def _flush(self):
        if self._n == 0:
            return
        arrays = []
        for name, typ in zip(self.schema.names, self.schema.types):
            vals = self._buf[name]
            arrays.append(pa.array(vals, type=typ))
        table = pa.Table.from_arrays(arrays, names=self.schema.names)
        # 行をまとめて 1 つの RowGroup として書く
        self.writer.write_table(table, row_group_size=self._n)
        # バッファをクリア
        self._buf = {name: [] for name in self.schema.names}
        self._n = 0

    def append(self, row: dict):
        for name in self.schema.names:
            self._buf[name].append(row.get(name))
        self._n += 1
        if self._n >= self.batch_rows:
            self._flush()

    def close(self):
        self._flush()
        self.writer.close()

def _colors32(cmap: Union[str, mcolors.Colormap]) -> np.ndarray:
    """
    指定カラーマップから 32 色の RGBA テーブル (32,4) を返す。
    ListedColormap でも LinearSegmentedColormap でも動く。
    """
    if isinstance(cmap, str):
        # lut=32 で 32 色に量子化された colormap を取得
        cmap = cm.get_cmap(cmap, 32)
    # 連続/離散を問わず確実に RGBA テーブルを得る
    table = cmap(np.linspace(0, 1, 32))  # shape (32, 4), 値は 0..1
    return np.asarray(table)

def color32(value: int,
            cmap: Union[str, mcolors.Colormap] = "viridis",
            *,
            as_hex: bool = False,
            clip: bool = False) -> Union[Color, str]:
    """
    0..31 の整数 value を 32色の離散カラーマップの色に対応させて返す。
    - cmap: 'viridis', 'plasma', 'turbo', 'jet' など
    - as_hex: True なら '#RRGGBB' を返す
    - clip: 範囲外を 0..31 に丸める（False なら ValueError）
    """
    v = int(value)
    if clip:
        v = max(0, min(31, v))
    elif not (0 <= v <= 31):
        raise ValueError(f"value must be in [0,31], got {value}")

    table = _colors32(cmap)           # (32, 4)
    rgba = tuple(map(float, table[v]))  # ensure plain floats
    return mcolors.to_hex(rgba) if as_hex else rgba

def color32_many(values: Iterable[int],
                 cmap: Union[str, mcolors.Colormap] = "viridis",
                 *,
                 as_hex: bool = False,
                 clip: bool = False) -> List[Union[Color, str]]:
    table = _colors32(cmap)
    out = []
    for value in values:
        v = int(value)
        if clip:
            v = max(0, min(31, v))
        elif not (0 <= v <= 31):
            raise ValueError(f"value must be in [0,31], got {value}")
        rgba = tuple(map(float, table[v]))
        out.append(mcolors.to_hex(rgba) if as_hex else rgba)
    return out

def pack_many_inverted(*flags: bool) -> tuple[int, int]:
    """
    flags[i] が True なら 0、False なら 1 を bit i に詰める。
    戻り値: (packed_int, width)
    """
    n = 0
    for i, f in enumerate(flags):
        bit = 0 if f else 1   # True→0, False→1
        n |= bit << i
    return n, len(flags)

def unpack_inverted(n: int, width: int) -> list[bool]:
    """
    pack_many_inverted で詰めた整数 n から元の bool 配列を復元。
    bit が 0 → True、1 → False に戻す。
    """
    return [((n >> i) & 1) == 0 for i in range(width)]

def build_posmap(pairs: List[Pair]) -> Dict[int, int]:
    """pos → byte の辞書にする（O(1) 参照用）"""
    return {pos: b & 0xFF for pos, b in pairs}

def read_bytes_after(posmap: Dict[int, int], marker_pos: int, n: int = 2) -> Optional[bytes]:
    """
    マーカー先頭位置 marker_pos の“直後”から n バイト読む。
    マーカーは2byte前提なので開始は marker_pos+2。
    どれか欠けていたら None。
    """
    start = marker_pos + 2
    out = []
    for i in range(n):
        p = start + i
        if p not in posmap:
            return None
        out.append(posmap[p])
    return bytes(out)

def extract_values_after_markers(
    pairs: List[Pair],
    marker_hits: Dict[str, List[int]],
    markers=("affa","faaf","fffa","fafa"),
    after_len: int = 2,
    hit_index: int = 0,  # 同一マーカーに複数ヒットがある場合の何個目を使うか
) -> Dict[str, Optional[str]]:
    """
    各マーカーの hit_index 番目の出現について、直後 after_len バイトを16進小文字で返す。
    不足/未ヒットなら None。
    """
    posmap = build_posmap(pairs)
    result: Dict[str, Optional[str]] = {}
    for key in markers:
        poss = marker_hits.get(key) or []
        if len(poss) <= hit_index:
            result[key] = None
            continue
        pos = poss[hit_index]
        bs = read_bytes_after(posmap, pos, n=after_len)
        result[key] = bs.hex() if bs is not None else None
    return result

def extract_timestamp_bytes(t1,t2,t3,debug):
    """Extract 48-bit timestamp from 6 bytes (t3, t2, t1)"""
    timestamp = (t3 << 32) | (t2 << 16) | t1
    if debug:
        print(f"t3={hex(t3)}, t2={hex(t2)}, t1={hex(t1)} => timestamp={timestamp}, ({hex(timestamp)})")
    return timestamp

def format_2byte_groups_colorized_from_pairs(pairs, colors, reset="\x1b[0m", last_byte_color=None):
    """
    pairs: [(pos:int, byte:int), ...]
    2Bごとに "aabb ccdd ..." を作り、colors の 4hex キーに一致すれば色付け。
    奇数末尾は 1B 表示。last_byte_color があれば末尾1Bに適用。
    """
    bs = bytes(b for _, b in pairs)
    out = []
    i = 0
    n = len(bs)
    while i + 1 < n:
        token = f"{bs[i]:02x}{bs[i+1]:02x}"
        c = colors.get(token.lower())
        out.append(f"{c}{token}{reset}" if c else token)
        i += 2
    if i < n:
        last = f"{bs[i]:02x}"
        out.append(f"{last_byte_color}{last}{reset}" if last_byte_color else last)
    return " ".join(out)

def gap_size_bytes_from_pairs(gap_pairs, start_pos=None, end_before=None):
    """位置付きgapから、[start_pos, end_before) のバイト数を数える。"""
    cnt = 0
    for pos, _b in gap_pairs:
        if start_pos is not None and pos < start_pos:
            continue
        if end_before is not None and pos >= end_before:
            continue
        cnt += 1
    return cnt

def gap_bytes_and_posmap(gap_pairs, start_pos=None, end_before=None):
    """
    gap_pairs をフィルタし、連続バイト列 bs と、bs[i] が元ストリームのどこに
    対応するかの posmap を返す。
      - bs: bytes
      - posmap: List[int]  (len(bs) と同じ長さで、各iの元オフセット)
    """
    bs_list = []
    posmap = []
    for pos, b in gap_pairs:
        if start_pos is not None and pos < start_pos:
            continue
        if end_before is not None and pos >= end_before:
            continue
        bs_list.append(b)
        posmap.append(pos)
    return bytes(bs_list), posmap

def find_markers_in_gap_pairs(gap_pairs, markers, start_pos=None, end_before=None):
    """
    gap_pairs の中から markers（2バイト値）を探して、元ストリームの絶対オフセットを返す。

    markers: 2バイトマーカーのリスト。要素は
             - int (例: 0xAFAF)
             - bytes (長さ2、例: b'\xAF\xAF')
             - str  (hex、例: 'afaf' or 'AF AF')
    戻り値: dict[marker_hex] = [abs_offset0, abs_offset1, ...]
            marker_hex は 'afaf' のような小文字16進4桁
    """
    # フィルタ済みのギャップをバイト列へ
    bs, posmap = gap_bytes_and_posmap(gap_pairs, start_pos, end_before)

    # マーカー正規化
    def norm_marker(m):
        if isinstance(m, int):
            m &= 0xFFFF
            return m.to_bytes(2, 'big')
        if isinstance(m, bytes):
            if len(m) != 2:
                raise ValueError("marker bytes must be length 2")
            return m
        if isinstance(m, str):
            h = m.replace(" ", "").replace("_", "")
            if len(h) != 4:
                raise ValueError("marker hex must be 4 hex chars for 2-byte marker")
            return bytes.fromhex(h)
        raise TypeError("unsupported marker type")
    needles = [(norm_marker(m), m) for m in markers]

    hits = {}
    for needle_bytes, orig in needles:
        key = needle_bytes.hex()  # 小文字 'afaf' など
        hits[key] = []
        # 重なりも検出する find-all
        i = 0
        n = len(needle_bytes)
        while True:
            j = bs.find(needle_bytes, i)
            if j == -1:
                break
            # 元ストリームの絶対開始オフセットは posmap[j]
            hits[key].append(posmap[j])
            i = j + 1
    return hits


def _bytes_from_pairs_interval(
    pairs: List[Pair],
    start_pos: int,
    end_before: int,
    *,
    fill_missing: Optional[int] = None,  # None=欠損で例外, 0..255ならその値で埋める
) -> bytes:
    """pairs から [start_pos, end_before) のバイト列を作る。"""
    if start_pos < 0 or end_before < start_pos:
        return b""
    # pos -> byte の辞書を作る（重複posがあれば後勝ち）
    posmap: Dict[int, int] = {p: (b & 0xFF) for p, b in pairs}
    out = bytearray()
    for p in range(start_pos, end_before):
        if p in posmap:
            out.append(posmap[p])
            # print(posmap[p],end=" ")
        else:
            if fill_missing is None:
                raise KeyError(f"missing byte at pos={p}")
            out.append(fill_missing & 0xFF)
    # print()
    return bytes(out)

def expand_10bit_units_from_pairs(
    pairs: List[Pair],
    start_pos: int,
    end_before: int,
    *,
    msb_first: bool = True,        # True=MSB優先（ビット7→0）、False=LSB優先（ビット0→7）
    fill_missing: Optional[int] = None,  # 区間内に欠損がある場合の埋め値（Noneは例外）
    pad_final_with_zeros: bool = False,  # 末尾に余りビットがあるとき 0 埋めで1ユニット出す
) -> List[int]:
    """
    pairs の [start_pos, end_before) をビット連結して 10bit ごとに分解し、各値(0..1023)のリストを返す。
    - msb_first=True: 先頭バイトの MSB が最初のビットとして流れる（ acc << 8 | b 方式 ）
    - msb_first=False: LSB が先に流れる（1bitずつ acc の下位側に積む方式）
    """
    data = _bytes_from_pairs_interval(pairs, start_pos, end_before, fill_missing=fill_missing)
    out: List[int] = []

    if msb_first:
        acc = 0       # 上位側に溜める
        acc_bits = 0  # acc に現在たまっているビット数
        for b in data:
            acc = ((acc << 8) | (b & 0xFF)) & ((1 << (acc_bits + 8)) - 1 if acc_bits + 8 <= 64 else (acc << 8) | (b & 0xFF))
            acc_bits += 8
            # 先頭(上位)から10ビットずつ切り出す
            while acc_bits >= 10:
                val = (acc >> (acc_bits - 10)) & 0x3FF
                out.append(val)
                acc_bits -= 10
                # 下位 acc_bits を残す（上位を捨てる）
                acc &= (1 << acc_bits) - 1 if acc_bits > 0 else 0
        if acc_bits > 0 and pad_final_with_zeros:
            # 余りを上位詰めで0埋め
            val = (acc << (10 - acc_bits)) & 0x3FF
            out.append(val)
    else:
        # LSB優先：ビット0→7の順でaccの下位側に積む。10ビット溜まったら下位10ビットを取り出す。
        acc = 0
        acc_bits = 0
        for b in data:
            for i in range(8):  # 0..7
                bit = (b >> i) & 1
                acc |= (bit << acc_bits)
                acc_bits += 1
                if acc_bits >= 10:
                    val = acc & 0x3FF
                    out.append(val)
                    acc >>= 10
                    acc_bits -= 10
        if acc_bits > 0 and pad_final_with_zeros:
            # 余りを下位詰めで0埋め
            val = acc & 0x3FF  # 0が自動で上位側に入る
            out.append(val)

    return out

def byte_to_hex_and_int(val: int):
    """
    val: 0..255 の整数
    戻り値: (hex_str: str, n: int)  ※hex_strは常に2桁小文字
    """
    try:
        if not isinstance(val, int):
            raise TypeError(f"val must be int, got {type(val).__name__}")
        if not (0 <= val <= 0xFF):
            raise ValueError(f"val out of range: {val}")

        b = bytes([val])          # 1バイトに詰める
        hex_str = b.hex()         # '00'..'ff' の2桁文字列（str）
        # ここで「英字は入らない前提」なら一応チェック（任意）
        if not hex_str.isdigit():
            # 前提に反するケース（例: '0a','ff'等）→ 例外またはスキップ
            raise ValueError(f"hex_str contains a-f: {hex_str}")

        n = int(hex_str, 16)      # 16進として整数化（※重要）
        return hex_str, n

    except Exception as e:
        # 必要に応じてログやデフォルト値
        # ここでは再送出
        raise

def scan_stream(path: str, header, footer, timestamp1, timestamp2, timestamp3, 
                chunk, limit, max_gap_bytes=None, footer_search_limit=60, 
                output1=None, output2=None, binary_checker_flag=False, event_check_flag=False):
    """
    afaf(=header) 検出 → 直後60B以内で fafa(=footer) を2B境界で探索。
    見つかれば afaf直後2B と fafa直後2B を比較して一致なら確定出力。
    見つからなければ、60B窓内で最初に出た afaf(=queued_head) で「窓満了時」に即フォールバック分割。
    queued_head が無い場合のみ、次に現れた afaf で分割。
    """

    if event_check_flag:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
        ax = axes.ravel()    

        for i in range(4):
            ax[i].set_title(f"pulse @ chip{i}")

    if output1 is not None:
        logger1 = SAMPADataParquetAppender(output1)

    if output2 is not None:
        logger2 = PulseParquetAppender(output2)


    COLORS = {
        "afaf": "\x1b[31m",  # red
        "affa": "\x1b[32m",  # green
        "fafa": "\x1b[34m",  # blue
        "fffa": "\x1b[35m",  # magenta
        "faaf": "\x1b[33m",  # yellow
    }
    RESET = "\x1b[0m"

    head0 = (header >> 8) & 0xFF
    head1 = header & 0xFF
    foot0 = (footer >> 8) & 0xFF
    foot1 = footer & 0xFF

    prev_shifted = [None] * 8
    have_prev_shifted = [False] * 8

    prev_raw = None
    prev_pos = -1
    total_bytes = 0

    # アクティブ・ブロック
    active = None
    # active = {
    #   'start': int,                  # afaf 先頭
    #   'align': int,                  # 検出に使ったビットアライン s（同アラインの afaf だけ候補にする）
    #   'pairs': [(pos, byte), ...],   # start+2 以降の生データ
    #   'first2': bytes|None,          # afaf直後の2B
    #   'footer_abs': int|None,        # 見つけた fafa 先頭（正常確定）
    #   'queued_head': int|None,       # 60B窓内で最初に見えた次の afaf
    #   'queued_used': bool,           # queued_head を使って再基準化済みか
    # }

    sample_block = []
    event_block = []

    def sample_builder(sample_block: list = [], row: Dict = {}, event_data_check_flag: bool = False):
        current_data = list(row.values())

        # print(current_data)

        if len(sample_block) == 0:
            sample_block.append(current_data)
            
        else:
            previous_chip = sample_block[len(sample_block)-1][3]
            previous_sample_index = sample_block[len(sample_block)-1][4]

            current_chip = current_data[3]
            current_sample_index = current_data[4]

            if ( previous_chip == current_chip ) and ( previous_sample_index < current_sample_index ):
                sample_block.append(current_data)

            else:
                # print(f"len {len(sample_block)}, pre({previous_chip}, {previous_sample_index}), cur({current_chip}, {current_sample_index})")
    
                transposed_sample_block = [list(col) for col in zip(*sample_block)]
                fadc_samples = [list(col) for col in zip(*transposed_sample_block[5])]
                
                sample_build_data = {
                    "chip" : transposed_sample_block[3][0],
                    "timestamp": transposed_sample_block[2],
                    "sample_index":  transposed_sample_block[4],
                    "samples_value" : fadc_samples
                }

                if output2 is not None:
                    logger2.append(sample_build_data)

                if len(event_block) == 0:
                    event_block.append(sample_build_data)
                
                else:
                    previous_event_block = event_block[len(event_block)-1]
                    current_event_block  = sample_build_data
                    
                    if abs( current_event_block['timestamp'][0] - previous_event_block['timestamp'][0] ) < 50:
                        event_block.append(sample_build_data)
                    
                    else:
                        # print(f"len {len(event_block)}")

                        if event_data_check_flag:
                            for data in event_block:
                                chip_number = data['chip']
                                sample_indices = data['sample_index']

                                ich = 0
                                
                                for sample_values in data['sample_value']:
                                    baseline = np.array(mode(sample_values))
                                    x = np.array(sample_indices)
                                    y = np.array(sample_values) - baseline
                                    ax[chip_number].plot(x, y, lw=1, alpha=0.5, marker="o", markersize=2, label=f"ch{ich}", color=color32(ich,"brg"))
                                    ich += 1
                            
                            for i in range(4):
                                ax[i].legend(loc='upper right', ncol=3, fontsize=6)

                            plt.show(block=False)   
                            plt.pause(0.01)   

                            for i in range(4):
                                ax[i].clear()
                                ax[i].set(xlim=(0, 64), ylim=(-300, 725))
                                ax[i].set_title(f"pulse @ chip{i}")     
                                ax[i].set_xlabel("Sample index")
                                ax[i].set_ylabel("Sample value - Base line(mode value)")     

                        event_block.clear()
                        event_block.append(sample_build_data)

                sample_block.clear()
                sample_block.append(current_data)
     
    def fmt_pairs_color(pairs):
        bs = bytes(b for _, b in pairs)
        out = []
        i = 0
        n = len(bs)
        while i + 1 < n:
            token = f"{bs[i]:02x}{bs[i+1]:02x}"
            c = COLORS.get(token.lower())
            out.append(f"{c}{token}{RESET}" if c else token)
            i += 2
        if i < n:
            out.append(f"{bs[i]:02x}")
        return " ".join(out)


    def emit_block(start_off, pairs, end_before_abs, tag_reason, dump_flag, event_data_check_flag):

        timestamp = None
        trimmed = [p for p in pairs if p[0] < end_before_abs]
        gap_str = fmt_pairs_color(trimmed)
        gap_size = len(trimmed)

        markers = [0xaffa, 0xfaaf, 0xfffa, 0xfafa]
        marker_hits = find_markers_in_gap_pairs(trimmed, markers)

        datasize_flag = ( gap_size == 58 )
        timestamp1_flag = ( len(marker_hits['affa'])>0 )
        timestamp2_flag = ( len(marker_hits['faaf'])>0 )
        timestamp3_flag = ( len(marker_hits['fffa'])>0 )
        footer_flag = ( len(marker_hits['fafa'])>0 )

        flag_binary, length = pack_many_inverted(datasize_flag, timestamp1_flag, timestamp2_flag, timestamp3_flag, footer_flag)  # → 2
        flag_debuger = 4 - timestamp1_flag - timestamp2_flag - timestamp3_flag - footer_flag

        vals_hex = extract_values_after_markers(pairs, marker_hits)
        vals_int = {k: (int(v,16) if v is not None else None) for k, v in vals_hex.items()}
    
        head_hex_col = f"{COLORS.get('afaf','')}{'afaf'}{RESET}" if COLORS.get('afaf') else "afaf"

        if 3 - timestamp1_flag - timestamp2_flag - timestamp3_flag == 0:
            timestamp = extract_timestamp_bytes(vals_int['fffa'], vals_int['faaf'], vals_int['affa'],False)

        chip_number_byte = read_bytes_after(build_posmap(pairs), start_off, n=1)
        sample_index_byte = read_bytes_after(build_posmap(pairs), start_off+1, n=1)

        vals= None

        if 2 - footer_flag - timestamp3_flag == 0:
            start_pos = marker_hits['fffa'][0]+4 
            end_before = marker_hits['fafa'][0] if len( marker_hits['fafa'] ) > 0 else end_before_abs
            vals = expand_10bit_units_from_pairs(pairs, start_pos, end_before, msb_first=True, fill_missing=0, pad_final_with_zeros=False)

        chip_number = int(chip_number_byte.hex(),16)
        sample_index= int(sample_index_byte.hex(),16)

        row = {
            "data_block": gap_size+2,
            "error_level": flag_debuger,
            "timestamp": timestamp,
            "chip": chip_number,
            "sample_index": sample_index,
            "samples_value":vals
        }
        
        if gap_size+2 == 60 and flag_debuger == 0:
            sample_builder(sample_block, row, event_data_check_flag)

        if output1 is not None:
            logger1.append(row)

        if dump_flag:
            if flag_debuger >= 0:
                print(
                    f"0x{start_off:08x}: {head_hex_col}"
                    + (f" {gap_str}" if gap_str else "")
                    + f"  ({gap_size+2}B)  {tag_reason}  error:{flag_debuger} timestamp:{timestamp} ({chip_number_byte.hex()}, {sample_index_byte.hex()})"
                )

    def rebase_active_to(start_off_new):
        """queued_head などを新しい開始にして active を再基準化"""
        nonlocal active
        new_pairs = [p for p in active['pairs'] if p[0] >= start_off_new + 2]
        new_first2 = None
        if len(new_pairs) >= 2:
            new_first2 = bytes([new_pairs[0][1], new_pairs[1][1]])
        active = {
            'start': start_off_new,
            'align': active['align'],
            'pairs': new_pairs,
            'first2': new_first2,
            'footer_abs': None,
            'queued_head': None,
            'queued_used': True,
        }

    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break

            for b in buf:
                if limit is not None and total_bytes >= limit:
                    return

                # --- 直前バイトをアクティブ窓へ蓄積 ---
                if prev_raw is not None and active is not None:
                    if prev_pos >= active['start'] + 2:
                        active['pairs'].append((prev_pos, prev_raw))
                        # first2
                        if active['first2'] is None and len(active['pairs']) >= 2:
                            active['first2'] = bytes([active['pairs'][0][1], active['pairs'][1][1]])

                        # 60B窓内 footer 探索
                        window_len = prev_pos - (active['start'] + 2) + 1
                        if active['first2'] is not None and window_len <= footer_search_limit:
                            bs = bytes(bb for _, bb in active['pairs'])
                            n = len(bs)
                            k = 0
                            while k + 3 < n:
                                if bs[k] == 0xFA and bs[k+1] == 0xFA:
                                    after_footer2 = bs[k+2:k+4]
                                    if after_footer2 == active['first2']:
                                        active['footer_abs'] = active['start'] + 2 + k + 4
                                        emit_block(active['start'], active['pairs'], active['footer_abs'],"ok:footer-in-60B",
                                                   dump_flag=binary_checker_flag, event_data_check_flag=event_check_flag)
                                        active = None
                                        break
                                k += 2
                        # 60B窓の満了チェック：footer 見つからず、queued_head があれば即フォールバック分割
                        if active is not None and active['footer_abs'] is None:
                            if window_len > footer_search_limit and active['queued_head'] is not None:
                                emit_block(active['start'], active['pairs'],
                                           end_before_abs=active['queued_head'],
                                           tag_reason="fallback:queued-head-on-expire",
                                           dump_flag=binary_checker_flag, event_data_check_flag=event_check_flag)
                                rebase_active_to(active['queued_head'])

                # prev_raw 初期化
                if prev_raw is None:
                    prev_raw = b
                    prev_pos = total_bytes
                    total_bytes += 1
                    continue

                curr_pos = total_bytes

                # --- 8アラインで整列 ---
                for s in range(8):
                    if s == 0:
                        shifted = prev_raw
                        out_pos = prev_pos
                    else:
                        mask = (1 << s) - 1
                        shifted = ((prev_raw >> s) | ((b & mask) << (8 - s))) & 0xFF
                        out_pos = prev_pos

                    if have_prev_shifted[s]:
                        # ヘッダ検出
                        if prev_shifted[s] == head0 and shifted == head1:
                            start_byte = out_pos - 1
                            if start_byte >= 0:
                                if active is None:
                                    # 新規ブロック開始
                                    active = {
                                        'start': start_byte,
                                        'align': s,
                                        'pairs': [],
                                        'first2': None,
                                        'footer_abs': None,
                                        'queued_head': None,
                                        'queued_used': False,
                                    }
                                else:
                                    # 既にブロック探索中
                                    if s != active['align']:
                                        # 別アラインの afaf は無視（誤検知で分割しない）
                                        pass
                                    else:
                                        window_len_now = prev_pos - (active['start'] + 2) + 1
                                        if active['footer_abs'] is None and window_len_now <= footer_search_limit:
                                            # 60B窓内：最初の afaf を queued_head に登録（分割はまだしない）
                                            if active['queued_head'] is None:
                                                active['queued_head'] = start_byte
                                        else:
                                            # 60B越え or 窓探索終わり：
                                            # queued_head が既に使われていれば、ここは「次の afaf」で通常分割
                                            if active['queued_head'] is None or active['queued_used']:
                                                emit_block(active['start'], active['pairs'],
                                                           end_before_abs=start_byte,
                                                           tag_reason="fallback:next-afaf",
                                                           dump_flag=binary_checker_flag,
                                                           event_data_check_flag=event_check_flag)
                                                active = {
                                                    'start': start_byte,
                                                    'align': s,
                                                    'pairs': [],
                                                    'first2': None,
                                                    'footer_abs': None,
                                                    'queued_head': None,
                                                    'queued_used': False,
                                                }
                                            else:
                                                # queued_head はあるが未使用で60Bは既に超えてる → もう使っているはずなので通常来ない分岐
                                                pass

                    prev_shifted[s] = shifted
                    have_prev_shifted[s] = True

                prev_raw = b
                prev_pos = curr_pos
                total_bytes += 1

    # EOF：未確定ブロックは出さない（孤立 afaf は無視）
    return

def common_options(func):
    # @click.option('--name', '-n', default='World', show_default=True, help='Your name')
    # @click.option('--date', '-d', type=click.DateTime(formats=["%Y-%m-%d"]), default=lambda: datetime.datetime.now(), show_default=lambda: datetime.datetime.now().strftime("%Y-%m-%d"), help="Date in YYYY-MM-DD format")
    # @click.option('--verbose', '-v', is_flag=True, help='verbose flag')
    # @click.option("--cols", '-c', type=int, default=6)
    # @click.option("--endian", '-e', type=click.Choice(["little", "big"]), default="big")
    # @click.option("--limit", '-l',type=int, default=None,help="ダンプする最大バイト数（未指定なら全体をダンプ）")
    @click.option("--limit", "-l", type=int, default=None, help="最大走査バイト数。指定しない場合は最後まで処理。")
    @click.option("--maxevt", "-m", type=int, default=-1, help="Plotする最大の件数")
    @click.option('--plot', '-p', is_flag=True, help='plot flag')
    @click.option('--binary', '-b', is_flag=True, help='plot flag')
    @click.option('--event', '-e', is_flag=True, help='plot event by event flag')
    @click.option('--decode', '-d', is_flag=True, help='decode flag')
    
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@common_options
def main(limit, plot, maxevt, binary, event, decode):

    toml_file_path = this_file_path  / "../../../parameters.toml"

    with open(toml_file_path, "r") as f:
        config = toml.load(f)

    format = config["format"]
    fileinfo = config["fileinfo"]

    HEADER_MARKER = format["header_marker"]# = 0xafaf
    FOOTER_MARKER = format["footer_marker"]# = 0xfafa
    T1_MARKER = format["t1_marker"]# = 0xfffa
    T2_MARKER = format["t2_marker"]# = 0xfaaf
    T3_MARKER = format["t3_marker"]# = 0xaffa  

    DATA = fileinfo["base_input_path"] + "/" + fileinfo["input_file_dir"] + "/" + fileinfo["input_file_name"] + ".bin"
    BASEOUTPUT = fileinfo["base_output_path"]  + "/" + fileinfo["input_file_name"] 
    OUTPUT1 = BASEOUTPUT + "_raw.parquet"
    OUTPUT2 = BASEOUTPUT + "_event.parquet"

    chunk = 1 << 10 

    if decode:
        scan_stream(DATA, HEADER_MARKER, 
                    FOOTER_MARKER, T1_MARKER , T2_MARKER, T3_MARKER, 
                    chunk, limit, output1=OUTPUT1, output2=OUTPUT2,
                    binary_checker_flag=binary, event_check_flag=event
        )

    if 0:
        spark = (SparkSession.builder.appName("ParquetFilter").getOrCreate())
        df = spark.read.parquet(OUTPUT1)
        df = (df
            .withColumn("chip_int", F.col("chip").cast("int"))                # or "long"
            .withColumn("sample_index_int", F.col("sample_index").cast("long"))
        )
        df.printSchema()
        print("Columns:", df.columns)

        df.show(10)

    if plot:  
        spark = (SparkSession.builder.appName("ParquetFilter").getOrCreate())
        df = spark.read.parquet(OUTPUT1)
        N = maxevt  # None なら全件、数値なら先頭N件だけ
        cols = ['data_block', 'error_level', 'timestamp', 'chip', 'sample_index', 'samples_value']

        data_block   = []
        error_level  = []
        timestamp    = []
        chip         = []
        sample_index = []
        max_samples  = []

        it = df.select(*cols).toLocalIterator()
        if N is not None:
            it = islice(it, N) 

        for row in it:
            db, err, ts, ch, ind, sval = row
            data_block.append(db)
            error_level.append(err)
            timestamp.append(ts)
            chip.append(ch)
            sample_index.append(ind)
            
            if ( sval is not None ) and ( len(sval) > 0 ) :
                max_samples.append(max(sval))

        fig = make_subplots(rows=2, cols=2, vertical_spacing=0.15, horizontal_spacing=0.1, subplot_titles=("data_block", "error_level", "sample values", "chip"))
        pau.add_sub_plot(fig,1,1,'1d',[data_block],['Data size','Counts'], xrange=[0,100,1],logs=[False, True])
        pau.add_sub_plot(fig,1,2,'1d',[error_level],['Error Status','Counts'], xrange=[0,4,1],logs=[False, True])
        pau.add_sub_plot(fig,2,1,'1d',[max_samples],['Max sample','Counts'], xrange=[0,1024,1],logs=[False, True])
        pau.add_sub_plot(fig,2,2,'1d',[chip],['Chip number','Counts'], xrange=[0,4,1])
        fig.update_layout(height=950, width=1400, title_text=f"File name:{fileinfo["input_file_name"]}.bin", showlegend=False)
        fig.show()

        fig = make_subplots(rows=2, cols=2, vertical_spacing=0.15, horizontal_spacing=0.1, subplot_titles=("timestamp", "sample_index", "chip", "error_level"))
        pau.add_sub_plot(fig,1,1,'scatter',[timestamp],['Number of data block','Timestamp'])
        pau.add_sub_plot(fig,1,2,'scatter',[sample_index],['Number of data block','Sample index'])
        pau.add_sub_plot(fig,2,1,'scatter',[chip],['Number of data block','Chip number'])
        pau.add_sub_plot(fig,2,2,'scatter',[error_level],['Number of data block','Error flag'])
        fig.update_layout(height=950, width=1400, title_text=f"File name:{fileinfo["input_file_name"]}.bin", showlegend=False)
        fig.show()

if __name__ == '__main__':
    main()
