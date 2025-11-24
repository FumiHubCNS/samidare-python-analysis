"""!
@file get_pininfo_fromXML.py
@version 1
@author FumiHubCNS
@date 2025-08-27T22:49:34+09:00
@brief template text
"""
import click
import pathlib
import datetime

this_file_path = pathlib.Path(__file__).parent

import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

NET_PAT = re.compile(r"^(IN|OUT)\$(\d+)$")
J_PART_PAT = re.compile(r"^J(\d+)$")

def extract_j_pinrefs_from_net(net_elem):
    """<net> 要素内の pinref から、part が J<number> のものだけを抽出"""
    results = []
    for pinref in net_elem.findall(".//pinref"):
        part = pinref.get("part", "")
        pin = pinref.get("pin", "")
        if J_PART_PAT.match(part):  # J で始まるコネクタだけ
            results.append((part, pin))
    return results

def parse_in_out_jpins(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 結果を {number: {"IN": [(J, pin), ...], "OUT": [(J, pin), ...]}} に格納
    by_num = defaultdict(lambda: {"IN": [], "OUT": []})

    for net in root.findall(".//net"):
        name = net.get("name", "")
        m = NET_PAT.match(name)
        if not m:
            continue
        kind, num = m.group(1), m.group(2)  # kind = IN / OUT
        jpins = extract_j_pinrefs_from_net(net)
        if jpins:
            by_num[num][kind].extend(jpins)

    return by_num

def print_report(by_num):
    # number 昇順でレポート
    for num in sorted(by_num, key=lambda x: int(x)):
        entries = by_num[num]
        for kind in ("IN", "OUT"):
            jpins = entries[kind]
            if not jpins:
                print(f"Number {num}  {kind} (no J-part pinrefs)")
                continue
            for part, pin in jpins:
                print(f"Number {num}  {kind} {part}  pin {pin}")

def main():
    xml_file = Path("/Users/fendo/Downloads/download/BBCHECK_SAMIDARE_ADAPTER.sch")  # 解析したいファイルに置き換え
    report = parse_in_out_jpins(xml_file)
    print_report(report)

if __name__ == '__main__':
    main()
