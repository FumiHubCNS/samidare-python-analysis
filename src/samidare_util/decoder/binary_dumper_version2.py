"""!
@file binary_dumper_version2.py
@version 1
@author Fumitaka ENDO
@date 2025-09-04T00:58:22+09:00
@brief Dump binary file
"""
import click
import pathlib
import datetime
import sys
import toml
from typing import Iterable
from typing import List, Optional

this_file_path = pathlib.Path(__file__).parent
sys.path.append(str(this_file_path.parent.parent.parent / "src"))


def format_2byte_groups_colorized_from_pairs(pairs, color_option, reset_option, last_byte_color=None):

    bs = bytes(b for _, b in pairs)
    out = []
    i = 0
    n = len(bs)
    while i + 1 < n:
        token = f"{bs[i]:02x}{bs[i+1]:02x}"
        c = color_option.get(token.lower())
        out.append(f"{c}{token}{reset_option}" if c else token)
        i += 2
    if i < n:  # 1バイト残り
        last = f"{bs[i]:02x}"
        if last_byte_color:
            out.append(f"{last_byte_color}{last}{reset_option}")
        else:
            out.append(last)
    return " ".join(out)


def gap_size_bytes_from_pairs(gap_pairs, start_pos=None, end_before=None):
    """
    gap_pairs: [(pos:int, byte:int), ...]
    start_pos: ここ以上の位置のみ数える（Noneなら先頭から）
    end_before: ここ"未満"の位置のみ数える（Noneなら末尾まで）
    戻り値: バイト数（=ギャップの長さ）
    """
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

def format_2byte_groups_from_pairs(pairs):
    """
    pairs: [(pos:int, byte:int), ...]
    を2バイトずつの"aaaa bbbb ... ee"形式のhex文字列にする。
    """
    bs = bytes(b for _, b in pairs)
    out = []
    i = 0
    n = len(bs)
    while i + 1 < n:
        out.append(f"{bs[i]:02x}{bs[i+1]:02x}")
        i += 2
    if i < n:
        out.append(f"{bs[i]:02x}")
    return " ".join(out)


def scan_stream(path: str, header, footer, timstamp1, timstamp2, timstamp3,
                chunk, limit, max_gap_bytes=None):
    
    color_option = {
    "afaf": "\x1b[31m",  # red
    "affa": "\x1b[32m",  # green
    "fafa": "\x1b[34m",  # blue
    "fffa": "\x1b[35m",  # magenta
    "faaf": "\x1b[33m",  # yellow
    }

    reset_option = "\x1b[0m"   

    
    # 2byteヘッダ/フッタに分解（フッタは未使用だが受け取っておく）
    head0 = (header >> 8) & 0xFF
    head1 = header & 0xFF

    foot0 = (footer >> 8) & 0xFF
    foot1 = footer & 0xFF

    prev_shifted = [None] * 8
    have_prev_shifted = [False] * 8

    prev_raw = None          # 直前の生バイト値
    prev_pos = -1            # 直前の生バイトのインデックス（0始まり）
    total_bytes = 0          # 読み進めた総バイト数

    # 直前ヒットと、その直後から次ヒット直前までの生データ（位置つき）
    prev_hit_offset = None   # int
    prev_hit_value  = None   # 0x0000..0xffff
    gap_start = None         # 前回ヒット開始+2（ギャップ開始位置）
    gap_buf = []             # [(pos, byte), ...]

    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break

            for b in buf:
                # limit 到達で終了（直前ヒットはここでflushしてから帰る）
                if limit is not None and total_bytes >= limit:
                    if prev_hit_offset is not None:
                        if 0:
                            print(f"リミットを超えるデータブロックです。")

                    return

                # ギャップ収集：前回ヒットが確定しているなら、ヒット直後(=gap_start)以降の位置だけ追加
                if prev_raw is not None and prev_hit_offset is not None:
                    if gap_start is not None and prev_pos >= gap_start:
                        gap_buf.append((prev_pos, prev_raw))
                        # （任意）メモリ上限：大きな間隔での暴走を避けたい場合
                        if max_gap_bytes is not None and len(gap_buf) > max_gap_bytes:
                            # 先頭を捨てる（必要ならリングバッファ等に置換可）
                            drop = len(gap_buf) - max_gap_bytes
                            if drop > 0:
                                gap_buf = gap_buf[drop:]

                if prev_raw is None:
                    prev_raw = b
                    prev_pos = total_bytes
                    total_bytes += 1
                    continue

                curr_pos = total_bytes  # b のインデックス

                # 全ビットオフセット(0..7)で整列後バイトを作って判定（MSB優先）
                for s in range(8):
                    if s == 0:
                        shifted = prev_raw
                        out_pos = prev_pos
                    else:
                        mask = (1 << s) - 1
                        shifted = ((prev_raw >> s) | ((b & mask) << (8 - s))) & 0xFF
                        out_pos = prev_pos

                    if have_prev_shifted[s]:
                        # ヘッダ検出（フッタ等は必要ならここに追加判定）
                        if prev_shifted[s] == head0 and shifted == head1:
                            start_byte = out_pos - 1  # ヘッダ先頭の元バイト位置
                            if start_byte >= 0:
                                hit_value = (prev_shifted[s] << 8) | shifted

                                if prev_hit_offset is None:
                                    # 初回ヒットは出力保留。ギャップ開始をこのヒット直後にセット
                                    prev_hit_offset = start_byte
                                    prev_hit_value  = hit_value
                                    gap_start = start_byte + 2
                                    gap_buf.clear()
                                else:
                                    start_pos  = prev_hit_offset + 2
                                    end_before = start_byte

                                    # gap のダンプ用に、次ヘッダを含めないようトリム
                                    trimmed = [p for p in gap_buf if p[0] < end_before]
                                    gap_str = format_2byte_groups_colorized_from_pairs(trimmed, color_option, reset_option)

                                    # prev の 2byte値自体もマップに従って色付け（該当なければ無色）
                                    prev_hex = f"{prev_hit_value:04x}"
                                    c = color_option.get(prev_hex.lower())
                                    prev_hex_colored = f"{c}{prev_hex}{reset_option}" if c else prev_hex

                                    # gap サイズ（バイト数）
                                    gap_size = gap_size_bytes_from_pairs(gap_buf, start_pos, end_before)

                                    # gap 内で探すマーカー（2byte, 16bit）
                                    markers = [timstamp1, timstamp2, timstamp3, footer]
                                    marker_hits = find_markers_in_gap_pairs(gap_buf, markers, start_pos, end_before)

                                    timestamp1_flag = ( len(marker_hits['affa'])>0 )
                                    timestamp2_flag = ( len(marker_hits['faaf'])>0 )
                                    timestamp3_flag = ( len(marker_hits['fffa'])>0 )
                                    footer_flag = ( len(marker_hits['fafa'])>0 )

                                    flag_debuger = 4 - timestamp1_flag - timestamp2_flag - timestamp3_flag - footer_flag 

                                    if footer_flag is True and timestamp3_flag is True:
                                        timstamp1_pos = marker_hits['fffa'][0] + 2 + 2 # 開始マーカー(2byte)は除外
                                        footer_pos    = marker_hits['fafa'][0]      # 終了マーカーは含めない

                                        trimmed_header_data = [p for p in gap_buf if p[0] < timstamp1_pos]
                                        trimmed_fadc_data   = [p for p in gap_buf if timstamp1_pos <= p[0] < footer_pos]
                                        trimmed_footer_data = [p for p in gap_buf if footer_pos <= p[0] < end_before]

                        
                                        gap_str_header = format_2byte_groups_colorized_from_pairs(trimmed_header_data, color_option, reset_option)
                                        gap_str_fadc_data = format_2byte_groups_colorized_from_pairs(trimmed_fadc_data, color_option, reset_option)
                                        gap_str_footer_data = format_2byte_groups_colorized_from_pairs(trimmed_footer_data, color_option, reset_option)

                                        gap_str = gap_str_header + " " + gap_str_fadc_data + " " + gap_str_footer_data
                                    

                                    # prev の 2byte値自体もマップに従って色付け（該当なければ無色）
                                    prev_hex = f"{prev_hit_value:04x}"
                                    c = color_option.get(prev_hex.lower())
                                    prev_hex_colored = f"{c}{prev_hex}{reset_option}" if c else prev_hex

                                    # gap サイズ（バイト数）
                                    gap_size = gap_size_bytes_from_pairs(gap_buf, start_pos, end_before)
                                    flag_debuger = 4 - timestamp1_flag - timestamp2_flag - timestamp3_flag - footer_flag 
                                    
                                    if flag_debuger > 0:
                                        print(
                                            f"0x{prev_hit_offset:08x}: {prev_hex_colored}"
                                            + (f" {gap_str}" if gap_str else "")
                                            + f"  (block size={gap_size+2}B)  error:{flag_debuger}"
                                    )

                                    # 次の区間準備
                                    prev_hit_offset = start_byte
                                    prev_hit_value  = hit_value
                                    gap_start = start_byte + 2
                                    gap_buf.clear()


                    prev_shifted[s] = shifted
                    have_prev_shifted[s] = True

                # 次へ
                prev_raw = b
                prev_pos = curr_pos
                total_bytes += 1

    # EOF：最後のヒットは、その後ろ（EOF直前）までのギャップを付けて出力
    if prev_hit_offset is not None:
        # gap のダンプ用に、次ヘッダを含めないようトリム
        trimmed = [p for p in gap_buf if p[0] < end_before]
        gap_str = format_2byte_groups_colorized_from_pairs(trimmed, color_option, reset_option)

        # prev の 2byte値自体もマップに従って色付け（該当なければ無色）
        prev_hex = f"{prev_hit_value:04x}"
        c = color_option.get(prev_hex.lower())
        prev_hex_colored = f"{c}{prev_hex}{reset_option}" if c else prev_hex

        # gap サイズ（バイト数）
        gap_size = gap_size_bytes_from_pairs(gap_buf, start_pos, end_before)

        # gap 内で探すマーカー（2byte, 16bit）
        markers = [timstamp1, timstamp2, timstamp3, footer]
        marker_hits = find_markers_in_gap_pairs(gap_buf, markers, start_pos, end_before)

        flag_debuger = 3 - ( len(marker_hits['affa'])>0 ) - ( len(marker_hits['faaf'])>0 ) - ( len(marker_hits['fffa'])>0 )
        
        timstamp1_pos = marker_hits['fffa'][0] + 2 + 2 # 開始マーカー(2byte)は除外
        footer_pos    = marker_hits['fafa'][0]      # 終了マーカーは含めない

        trimmed_header_data = [p for p in gap_buf if p[0] < timstamp1_pos]
        trimmed_fadc_data   = [p for p in gap_buf if timstamp1_pos <= p[0] < footer_pos]
        trimmed_footer_data = [p for p in gap_buf if footer_pos <= p[0] < end_before]


        gap_str_header = format_2byte_groups_colorized_from_pairs(trimmed_header_data, color_option, reset_option)
        gap_str_fadc_data = format_2byte_groups_colorized_from_pairs(trimmed_fadc_data, color_option, reset_option)
        gap_str_footer_data = format_2byte_groups_colorized_from_pairs(trimmed_footer_data, color_option, reset_option)

        gap_str = gap_str_header + " " + gap_str_fadc_data + " " + gap_str_footer_data

        # prev の 2byte値自体もマップに従って色付け（該当なければ無色）
        prev_hex = f"{prev_hit_value:04x}"
        c = color_option.get(prev_hex.lower())
        prev_hex_colored = f"{c}{prev_hex}{reset_option}" if c else prev_hex

        # gap サイズ（バイト数）
        gap_size = gap_size_bytes_from_pairs(gap_buf, start_pos, end_before)
        # if gap_size%2 ==1:
        print(
            f"0x{prev_hit_offset:08x}: {prev_hex_colored}"
            + (f" {gap_str}" if gap_str else "")
            + f"  (block size={gap_size+2}B)  error:{flag_debuger}"
        )


def common_options(func):
    @click.option('--name', '-n', default='World', show_default=True, help='Your name')
    @click.option('--date', '-d', type=click.DateTime(formats=["%Y-%m-%d"]), default=lambda: datetime.datetime.now(), show_default=lambda: datetime.datetime.now().strftime("%Y-%m-%d"), help="Date in YYYY-MM-DD format")
    @click.option('--verbose', '-v', is_flag=True, help='verbose flag')
    # @click.option("--cols", '-c', type=int, default=6)
    # @click.option("--endian", '-e', type=click.Choice(["little", "big"]), default="big")
    # @click.option("--limit", '-l',type=int, default=None,help="ダンプする最大バイト数（未指定なら全体をダンプ）")
    @click.option("--limit", "-l", type=int, default=None, help="最大走査バイト数。指定しない場合は最後まで処理。")
    # @click.option("--offset", '-o', type=str, default="0")
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@common_options
def main(name, date, verbose, limit):
    if verbose:
        click.echo(f"[VERBOSE MODE] Hello {name}, date: {date.strftime('%Y-%m-%d')}")
    else:
        click.echo(f"Hello {name}! (Date: {date.strftime('%Y-%m-%d')})")

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

    BASE = fileinfo["base_input_path"] + "/" + fileinfo["input_file_dir"] + "/"
    FILE = fileinfo["input_file_name"] + ".bin"

    DATA = BASE+FILE


    chunk = 1 << 10 #= 1024

    scan_stream(DATA, HEADER_MARKER, FOOTER_MARKER, T1_MARKER , T2_MARKER, T3_MARKER, chunk, limit)

if __name__ == '__main__':
    main()
