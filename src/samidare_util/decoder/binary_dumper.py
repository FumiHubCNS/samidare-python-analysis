"""!
@file binary_dumper.py
@version 1
@author Fumitaka ENDO
@date 2025-09-03T23:00:49+09:00
@brief Dump binary file
"""
import click
import pathlib
import datetime
import sys
import toml

this_file_path = pathlib.Path(__file__).parent
sys.path.append(str(this_file_path.parent.parent.parent / "src"))

def colorize(word: str, color_list) -> str:
    return f"{color_list[word]}{word}{"\x1b[0m"}" if word in color_list else word

def dump_stream(path, cols=6, endian="big", chunk=4096, color_list=None, limit=None, dump_offset="0"):
    
    if isinstance(dump_offset, str):
        dump_offset = int(dump_offset, 16)

    with open(path, "rb") as f:
        col = 0
        leftover = b""
        total_read = 0  # バイト数カウンター
        previous_word = None
        fafa_mode = False  # ← 追加：fafa直後の処理フラグ
        current_offset = 0

        while True:
            if limit is not None:
                # limit を超えないように読むサイズを調整
                remaining = limit - total_read
                if remaining <= 0:
                    break
                read_size = min(chunk, remaining)
            else:
                read_size = chunk

            buf = f.read(read_size)
            if not buf:
                break
            total_read += len(buf)

            buf = leftover + buf
            if len(buf) % 2 == 1:
                leftover = buf[-1:]
                buf = buf[:-1]
            else:
                leftover = b""

            for i in range(0, len(buf), 2):
                if current_offset < dump_offset:
                    current_offset += 2
                    continue

                w = buf[i:i+2]
        
                if endian == "little":
                    word = f"{w[1]:02x}{w[0]:02x}"
                else:
                    word = f"{w[0]:02x}{w[1]:02x}"

                if word == "afaf":
                    if col != 0:
                        print()
                    print(f"{current_offset:08x}: ", end=" ")
                    col = 0

                if word == "fafa":
                    fafa_mode = True  # 次の1語を特別扱い
                elif fafa_mode:
                    print(f"{colorize(word, color_list)} ({int(word[0:2],16):>2}, {int(word[2:4],16):>3})", end=" ")  # 特別処理
                    fafa_mode = False
                    col += 1
                    if col >= cols:
                        print()
                        col = 0
                    continue  # 処理済みなので次へ

                print(colorize(word, color_list), end=" ")
                
                col += 1
                current_offset += 2
                
                if col >= cols:
                    print()
                    print(f"{current_offset:08x}: ", end=" ")
                    col = 0


        if leftover and (limit is None or total_read < limit):
            # 端数処理（1バイトだけ）
            word = f"{leftover[0]:02x}00"
            print(colorize(word))


def common_options(func):
    @click.option('--name', '-n', default='World', show_default=True, help='Your name')
    @click.option('--date', '-d', type=click.DateTime(formats=["%Y-%m-%d"]), default=lambda: datetime.datetime.now(), show_default=lambda: datetime.datetime.now().strftime("%Y-%m-%d"), help="Date in YYYY-MM-DD format")
    @click.option('--verbose', '-v', is_flag=True, help='verbose flag')
    @click.option("--cols", '-c', type=int, default=6)
    @click.option("--endian", '-e', type=click.Choice(["little", "big"]), default="big")
    @click.option("--limit", '-l',type=int, default=None,help="ダンプする最大バイト数（未指定なら全体をダンプ）")
    @click.option("--offset", '-o', type=str, default="0")
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@common_options
def main(name, date, verbose, cols, endian, limit, offset):
    if verbose:
        click.echo(f"[VERBOSE MODE] Hello {name}, date: {date.strftime('%Y-%m-%d')}")
    else:
        click.echo(f"Hello {name}! (Date: {date.strftime('%Y-%m-%d')})")

    toml_file_path = this_file_path  / "../../../parameters.toml"

    with open(toml_file_path, "r") as f:
        config = toml.load(f)

    format = config["format"]
    fileinfo = config["fileinfo"]

    # HEADER_MARKER = format["header_marker"]# = 0xafaf
    # FOOTER_MARKER = format["footer_marker"]# = 0xfafa
    # T1_MARKER = format["t1_marker"]# = 0xfffa
    # T2_MARKER = format["t2_marker"]# = 0xfaaf
    # T3_MARKER = format["t3_marker"]# = 0xaffa  

    COLORS = {
        "afaf": "\x1b[31m",  # red
        "affa": "\x1b[32m",  # green
        "fafa": "\x1b[34m",  # blue
        "fffa": "\x1b[35m",  # magenta
        "faaf": "\x1b[33m",  # yellow
    }

    BASE = fileinfo["base_input_path"] + "/" + fileinfo["input_file_dir"] + "/"
    FILE = fileinfo["input_file_name"] + ".bin"

    DATA = BASE+FILE

    dump_stream(DATA, cols, endian, 4096, COLORS, limit, offset)

if __name__ == '__main__':
    main()
