"""!
@file chk_samidare_root.py
@version 1
@author Fumitaka ENDO
@date 2025-07-18T15:10:23+09:00
@brief check root file
"""
import argparse
import pathlib
import uproot
import pandas as pd 
import matplotlib.pyplot as plt
import awkward as ak
import numpy as np
from collections import defaultdict
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from collections import Counter
import time
import catm_lib as cat
import samidare_lib as tw

this_file_path = pathlib.Path(__file__).parent

def find_threshold_crossings(lst, threshold):
    above = False
    crossings_up = []
    crossings_down = []

    for i, val in enumerate(lst):
        if not above and val > threshold:
            crossings_up.append(i)
            above = True
        elif above and val <= threshold:
            crossings_down.append(i)
            above = False

    return crossings_up, crossings_down

def process_event(event, chip, ch, samples, raw_max, max_sample=60):
    id = ch + 32 * chip
    raw = samples[0:max_sample]
    counter = Counter(raw)
    most_common = counter.most_common(1)[0][0]
    
    raw_signed = np.array(raw, dtype=np.int32)
    y = raw_signed - int(most_common)

    return id, y


def self_trigger():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", help="input file path", type=str, default="/Users/fendo/Work/Data/root/sampa-minitpc-test/test_TPC_001.root")
    #parser.add_argument("-i", "--input", help="input file path", type=str, default="/Users/fendo/Work/Data/root/sampa-minitpc-test/test_TPC_1kHz_001.root")  
    parser.add_argument("-n", "--maxn", help="event number for loading data using ak", type=int, default=5000)  
    
    # parser.add_argument("-f", "--flag", help="sample flag", action="store_true")

    args = parser.parse_args()

    rootfile = args.input
    max_number = args.maxn
    # flag = args.flag

    idmap = load_map()
    print(idmap.head(10))

    file = uproot.open(rootfile)
    tree = file["tree"] 
    
    branch_list = tree.keys()
    print(branch_list)
    arrays = tree.arrays(entry_stop=max_number, library="ak")

    ch_accepted_waveforms = defaultdict(list)
    ch_all_waveforms = defaultdict(list)
    
    maximum_sample=60

    print("number of loop:",len(arrays["samples"]))
    maxloop = max_number
    
    start = time.time()

    for i in range(len(arrays["samples"])):
        event   = arrays["event"][i]
        chip    = arrays["chip"][i]
        ch      = arrays["ch"][i]
        samples = arrays["samples"][i]
        raw_max = arrays["raw_max_sample"][i]

        raw = samples[0:maximum_sample]
        id = ch+32*chip

        x = range(len(raw))
        
        counter = Counter(raw)

        most_common = counter.most_common(1)[0]  
        raw_signed = raw.to_numpy().astype(np.int32)
        baseline_signed = int(most_common[0])  
        y = raw_signed - baseline_signed

        threshold = 200
        above = y > threshold
        crossings_up = np.where((~above[:-1]) & (above[1:]))[0] + 1
        crossings_down = np.where((above[:-1]) & (~above[1:]))[0] + 1

        ch_all_waveforms[id].append(y)
        
        flag5 = False if len(crossings_up)*len(crossings_down) == 1 else True

        if flag5:
            continue

        taily = y[10:60]
        lower = -50
        upper = 50
        flag1 = np.all((taily >= lower) & (taily <= upper))

        pulse = y[0:10]
        lower = -10
        upper = 1000 
        flag2 = np.all((pulse >= lower) & (pulse <= upper))

        flag3 = True if pulse.max() > 100 else False
        flag4 = True if pulse.sum() > 400 else False

        flag6 = True if crossings_down[0] - crossings_up[0] >=4 else False

        
        if flag1 and flag2 and flag3 and flag4 and flag6:

            diffs = -1*np.diff(pulse)
            threshold = 200
            flag7 = np.abs(diffs) <= threshold

            if np.all(flag7):
                ch_accepted_waveforms[id].append(y)
        
        if i > maxloop:#50000:  
            break
    
    end = time.time()

    print(f"実行時間: {end - start:.3f} 秒")

    if 1:
        rows=11
        cols=12
        titles = []

        plotnumflgs = [0] * 200

        for i in range (rows*cols):
            if i <128:
                titles.append(f"Ch:{i}, pad:{idmap.tpcID[i]}-{idmap.padID[i]}")
            else:
                titles.append(f"Ch:{i}")

        ref = ch_accepted_waveforms
        # ref = ch_all_waveforms
            
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles)

        for ch, waveforms in ref.items():
            for wf in waveforms:
                row = (ch // cols) + 1
                col = (ch % cols) + 1
                x = np.arange(len(wf))
                if row > rows or col > cols:
                    print(f"⚠️ ch={ch} → (row={row}, col={col}) は範囲外。スキップします")
                    continue

                # if plotnumflgs[ch] == 0:
                if plotnumflgs[ch] < 3:
                    fig.add_trace(go.Scatter(x= x, y= wf), row=row, col=col)
                plotnumflgs[ch] += 1

        fig.update_layout(title = f"Pulse Shape (self-triger mode) gated data, loop:{maxloop}", showlegend=False)
        fig.update_annotations(font=dict(family="Helvetica", size=8))     
        # fig.update_yaxes(range=[-500, 500])#, row=row, col=col)
        fig.show()


def clock_trigger():

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", help="input file path", type=str, default="/Users/fendo/Work/Data/root/sampa-minitpc-test/test_TPC_1kHz_001.root")  
    parser.add_argument("-n", "--maxn", help="event number for loading data using ak", type=int, default=50000)  
    
    # parser.add_argument("-f", "--flag", help="sample flag", action="store_true")

    args = parser.parse_args()

    rootfile = args.input
    max_number = args.maxn
    # flag = args.flag

    file = uproot.open(rootfile)
    tree = file["tree"] 
    
    branch_list = tree.keys()
    print(branch_list)

    ch_accepted_waveforms = defaultdict(list)
    ch_all_waveforms = defaultdict(list)
    
    maximum_sample=60
    step_number = 10000

    print(f"max count : {max_number}, number of loop : {max_number//step_number}, number of 1 loop : {step_number}")

    start = time.time()

    for j in range (max_number//step_number):

        start_i = time.time()
            
        arrays = tree.arrays(entry_start=j*step_number ,entry_stop=(j+1)*step_number, library="ak")

        for i in range(len(arrays["samples"])):
            event   = arrays["event"][i]
            chip    = arrays["chip"][i]
            ch      = arrays["ch"][i]
            samples = arrays["samples"][i]
            raw_max = arrays["raw_max_sample"][i]

            raw = samples[0:maximum_sample]
            id = ch+32*chip

            x = range(len(raw))
            
            counter = Counter(raw)

            most_common = counter.most_common(1)[0]  
            raw_signed = raw.to_numpy().astype(np.int32)
            baseline_signed = int(most_common[0])  
            y = raw_signed - baseline_signed

            ch_all_waveforms[id].append(y)
        
        end_i = time.time()

        print(f"loop {j}: {end_i - start_i:.3f} 秒")
    
    end = time.time()

    print(f"実行時間: {end - start:.3f} 秒")

    if 1:
        rows=11
        cols=12
        titles = []

        plotnumflgs = [0] * 200

        for i in range (rows*cols):
            titles.append(f"Ch:{i}")
        
        # ref = ch_accepted_waveforms
        ref = ch_all_waveforms
            
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles)

        for ch, waveforms in ref.items():
            for wf in waveforms:
                row = (ch // cols) + 1
                col = (ch % cols) + 1
                x = np.arange(len(wf))
                if row > rows or col > cols:
                    print(f"⚠️ ch={ch} → (row={row}, col={col}) は範囲外。スキップします")
                    continue

                # if plotnumflgs[ch] == 0:
                if plotnumflgs[ch] < 3:
                    fig.add_trace(go.Scatter(x= x, y= wf), row=row, col=col)
                plotnumflgs[ch] += 1

        fig.update_layout(title = f"Pulse Shape (self-triger mode) gated data, loop:{max_number}", showlegend=False)
        fig.update_annotations(font=dict(family="Helvetica", size=8))     
        # fig.update_yaxes(range=[-500, 500])#, row=row, col=col)
        fig.show()

def load_map(inputpath='/Users/fendo/Work/Program/uv-python/test-work/prm/minitpc-samidare-test-at-ribf-2025-07/minitpc.map'):
    df = pd.read_csv(inputpath, comment='#', sep=',')
    return df

if __name__ == "__main__":

    idmap = load_map()
    # invalid_idmap = idmap[idmap.padID=="G"]
    # valid_idmap = idmap[idmap.padID!="G"]
    # print(invalid_idmap.head(10)) 
    # print(valid_idmap.head(20)) 

    # clock_trigger()
    # self_trigger()

    pad1 = tw.padinfo.get_tpc_info(45)
    pad2 = tw.padinfo.get_tpc_info(141.5)
    
    print(len(pad1.centers))
