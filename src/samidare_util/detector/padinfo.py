"""!
@file padinfo.py
@version 1
@author Fumitaka ENDO
@date 2025-07-25T20:10:19+09:00
@brief get pad configuration 
"""
import argparse
import pathlib
import catmlib as cat
import numpy as np
import copy
import catmlib.util.catmviewer as catview
from math import isclose, isnan
from typing import Sequence, Union, Optional

this_file_path = pathlib.Path(__file__).parent

Number = Union[int, float]

def find_index(seq: Sequence[Number], target: Number, *, tol: Optional[float] = None) -> int:
    """
    seq 内で target に最初に一致する要素のインデックスを返す。
    - tol を指定すると「絶対誤差 tol 以内」で一致とみなす（浮動小数向け）。
    - 見つからなければ ValueError を送出。
    """
    if tol is None:
        # NaN 同士を一致とみなしたい場合の特別扱い
        if isinstance(target, float) and isnan(target):
            for i, v in enumerate(seq):
                if isinstance(v, float) and isnan(v):
                    return i
        else:
            for i, v in enumerate(seq):
                if v == target:
                    return i
    else:
        for i, v in enumerate(seq):
            try:
                if isclose(float(v), float(target), rel_tol=0.0, abs_tol=tol):
                    return i
            except (TypeError, ValueError):
                # 数値化できない要素はスキップ
                continue

    raise ValueError(f"見つかりませんでした: {target}")

def get_opopsite_id(ref):
  val1 = ref%20
  val2 = ref//20
  return 19 - val1 + 20 * val2


def get_tpc_info(zoffset=0, oposite_flag=True):  

  base_padinfo = cat.readoutpad.basepad.generate_regular_n_polygon(3, 2.75, 90,False)

  pad = cat.readoutpad.basepad.TReadoutPadArray()
  pad.add_basepad(base_padinfo)
  pad_distance = 3
  gid = 0
  nmax=21
  offset = 15.75

  if oposite_flag:
    for i in range(nmax+1):
      if i >1:
        pad.add_pads([(i-1.5)*pad_distance/2-offset, 0, pad_distance/np.sqrt(3)/2*((i)%2)+zoffset], 0, 0, 180*((i)%2), 0,  gid)
        gid += 1
    for i in range(nmax):
      if i >0:
        pad.add_pads([((i+0))*pad_distance/2-offset, 0, pad_distance/np.sqrt(3)/2*((i-1)%2) + np.sqrt(3)*pad_distance/2+zoffset], 0, 0, 180*((i-1)%2), 0,  gid)
        gid += 1
    for i in range(nmax):
      if i < nmax-1:
        pad.add_pads([(i+1.5)*pad_distance/2-offset, 0, pad_distance/np.sqrt(3)/2*((i)%2) + 2*np.sqrt(3)*pad_distance/2+zoffset] , 0, 0, 180*((i)%2), 0,  gid)
        gid += 1
  else:
    for i in range(nmax+1):
      if i >1:
        pad.add_pads([(i-0.5)*pad_distance/2-offset, 0, pad_distance/np.sqrt(3)/2*((i-1)%2)+zoffset], 0, 0, 180*((i-1)%2), 0, get_opopsite_id(gid))
        gid += 1
    for i in range(nmax):
      if i >0:
        pad.add_pads([((i+0))*pad_distance/2-offset, 0, pad_distance/np.sqrt(3)/2*((i)%2) + np.sqrt(3)*pad_distance/2+zoffset], 0, 0, 180*((i)%2), 0, get_opopsite_id(gid))
        gid += 1
    for i in range(nmax):
      if i < nmax-1:
        pad.add_pads([(i+0.5)*pad_distance/2-offset, 0, pad_distance/np.sqrt(3)/2*((i-1)%2) + 2*np.sqrt(3)*pad_distance/2+zoffset] , 0, 0, 180*((i-1)%2), 0, get_opopsite_id(gid))
        gid += 1

  return pad



def marge_padinfos(pad1:cat.readoutpad.basepad.TReadoutPadArray, pad2: cat.readoutpad.basepad.TReadoutPadArray):

  obj = copy.deepcopy(pad1)
  gid = max(pad1.ids)

  offset = gid + 1 
  for i in range(len(pad2.ids)):
    polygon_new = pad2.pads[i]
    id = offset + pad2.ids[i]

    obj.pads.append(polygon_new)
    obj.ids.append(id)
    obj.centers.append(np.mean(polygon_new, axis=0))
    obj.charges.append(0)

  return obj

def main():

    offset = -3.031088913245535
    pad1 = get_tpc_info(offset+45)
    pad2 = get_tpc_info(offset+136.5,False)
    tpcs = marge_padinfos(pad1,pad2)

    cehck_list = tpcs.ids
    bins, colors = catview.get_color_list(cehck_list, cmap_name="rainbow", fmt="hex")
    color_array  = catview.get_color_array(cehck_list,bins,colors)
    tpcs.show_pads(check_id=True, check_size=5, plot_type='map',color_map=color_array, check_data=tpcs.ids)

if __name__ == '__main__':
  main()