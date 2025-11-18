from pyspark.sql import SparkSession
import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import toml
import pathlib
import sys
import re 
from typing import Tuple, Dict, Optional

this_file_path = pathlib.Path(__file__).parent
sys.path.append(str(this_file_path.parent.parent.parent / "src"))

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

def get_2darray_from_3darray(data3):
    data2 = [waveform for group in data3 for waveform in group]
    return data2

def get_1darray_from_2darray(data2):
    data1 = [item for sublist in data2 for item in sublist]
    return data1

def get_2dindex_array(data):
    index = [[j for j in range(len(row))] for row in data]
    return index

def get_1dindex_array(data):
    index = [j for j in range(len(data))]
    return index


def extract_valid_segments(chips, chans, timestamps, samples, times,
                           threshold_up=100, threshold_down=100,
                           pre_samples=12, post_samples=12,
                           baseline_samples=10):
    filtered_chips = []
    filtered_chans = []
    filtered_timestamps = []
    filtered_samples = []
    filtered_times = []

    for i in range(len(samples)):
        waveform = samples[i]
        if not waveform or len(waveform) < 20:
            continue

        front = waveform[:11]
        back = waveform[-11:]
        combined = front + back
        if not combined:
            continue
        baseline_1 = sum(combined) / len(combined)
        corrected_waveform = [v - baseline_1 for v in waveform]

        rise_index = next((j for j, v in enumerate(corrected_waveform) if v >= threshold_up), None)
        if rise_index is None:
            continue

        fall_index = next(
            (j for j in range(rise_index, len(corrected_waveform)) if corrected_waveform[j] < threshold_down),
            len(corrected_waveform) - 1
        )

        start = max(0, rise_index - pre_samples)
        end = min(len(corrected_waveform), fall_index + post_samples)
        segment = corrected_waveform[start:end]

        if len(segment) < baseline_samples:
            continue

        baseline_2 = sum(segment[:baseline_samples]) / baseline_samples
        final_segment = [v - baseline_2 for v in segment]

        filtered_chips.append(chips[i])
        filtered_chans.append(chans[i])
        filtered_timestamps.append(timestamps[i])
        filtered_samples.append(final_segment)
        filtered_times.append(times[i])

    return (filtered_chips, filtered_chans, filtered_timestamps, filtered_samples, filtered_times)

def get_maxsample_timing(data):
    indecies = []
    for i in range(len(data)):
        lst = data[i]
        if not lst:
            indecies.append(None)  # 空リストの場合
        else: 
            indecies.append(lst.index(max(lst)))
    return indecies

def get_pulse_info(data):
    counts = []
    max_samples = []
    charges = []

    for j in range(len(data)):
        counts.append(len(data[j]))
        max_samples.append(max(data[j]))
        charges.append(sum(data[j]))
    
    return (counts, max_samples, charges)


def build_event_usung_dt(input, dt, debug=False):
    INPUT_PARQUET_PATH = input
    MAX_DELTA = dt

    spark = SparkSession.builder.appName("IterativeParquetReader").getOrCreate()
    df = spark.read.parquet(INPUT_PARQUET_PATH)

    grouped_samples = []      
    grouped_timestamp = [] 
    grouped_chip = []   
    grouped_channel = []  

    current_samples = []    
    current_timestamp = []    
    current_chip = []   
    current_channel = [] 

    reference_ts = None 
    debug_counts = 0

    for row in df.toLocalIterator():
        ts = row["timestamp"]
        sample = row["samples"]
        chips = row["chip"]
        channels = row["channel"]

        if reference_ts is None:
            reference_ts = ts

        if debug:
            if debug_counts%32 == 0 and debug_counts < 620:
                print(f"evt id: {debug_counts}, timestamp:{ts}, {chips}, {channels}")

        if abs(ts - reference_ts) <= MAX_DELTA:
            current_samples.append(sample)
            current_timestamp.append(ts)
            current_chip.append(chips)
            current_channel.append(channels)
        else:
            if current_samples:
                grouped_samples.append(current_samples)
                grouped_timestamp.append(current_timestamp)
                grouped_chip.append(current_chip)
                grouped_channel.append(current_channel)

            current_samples = [sample]
            current_timestamp = [ts]
            current_chip = [chips]
            current_channel = [channels]

            reference_ts = ts

        debug_counts = debug_counts + 1

    if current_samples:
        grouped_samples.append(current_samples)
        grouped_timestamp.append(current_timestamp)
        grouped_chip.append(current_chip)
        grouped_channel.append(current_channel)

    return ( grouped_samples, grouped_timestamp, grouped_chip, grouped_channel )

def find_pulse(grouped_samples, grouped_timestamp, grouped_chip, grouped_channel, debug):

    filtered_grouped_samples = []      
    filtered_grouped_timestamp = [] 
    filtered_grouped_chip = []   
    filtered_grouped_channel = []  
    filtered_grouped_event_id = []
    filtered_grouped_times = []
    filtered_grouped_counts = []
    filtered_grouped_max_sample = []
    filtered_grouped_charge = []
    filtered_grouped_total_charge = []
    event_id = 0

    for i in range(len(grouped_samples)):

        max_sapmple_time = get_maxsample_timing(grouped_samples[i])

        if debug:
            if i < 20:
                print(f"evt: {i}, timestamp {grouped_timestamp[i][0]}")

        filtered_chips, filtered_chans, filtered_timestamps, filtered_samples, filtered_times = \
            extract_valid_segments(grouped_chip[i], grouped_channel[i], grouped_timestamp[i], grouped_samples[i], max_sapmple_time,\
                threshold_up=20, threshold_down=20, pre_samples=12, post_samples=12, baseline_samples=10)
        
        filtered_counts, filtered_max_samples, filtered_charges = get_pulse_info(filtered_samples)
        
        if len(filtered_samples) > 0:
            filtered_grouped_samples.append(filtered_samples)
            filtered_grouped_timestamp.append(filtered_timestamps)
            filtered_grouped_chip.append(filtered_chips)
            filtered_grouped_channel.append(filtered_chans)
            filtered_grouped_event_id.append(event_id)
            filtered_grouped_times.append(filtered_times)

            filtered_grouped_counts.append(filtered_counts)
            filtered_grouped_max_sample.append(filtered_max_samples)
            filtered_grouped_charge.append(filtered_charges)
            filtered_grouped_total_charge.append(sum(filtered_charges))
            
            event_id = event_id + 1

            if debug:
                if event_id <20:
                    for k in range(len(filtered_chans)):
                        print(f"evnt id: {event_id}, timstamp: {filtered_timestamps[k]} ({filtered_chips[k]}, {filtered_chans[k]})")

    return ( filtered_grouped_samples, filtered_grouped_timestamp, filtered_grouped_chip, filtered_grouped_channel, filtered_grouped_event_id, filtered_grouped_times, \
            filtered_grouped_counts, filtered_grouped_max_sample, filtered_grouped_charge, filtered_grouped_total_charge )


# counts, xedges, yedges = hist2d_counts(x_clean, y_clean, bins=bins, xrange=xrange, yrange=yrange)
def get_np_histogram2d(
    data: list = None,
    bins: list = None,
    xrange: list = None,
    yrange: list = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2Dヒストの生カウントとエッジを返す（plotly では z=counts.T を使う想定）
    """
    if data is None:
        return None
    
    if xrange is None:
        xrange = []

    if yrange is None:
        yrange = []
    
    x = data[0]
    y = data[1]

    x = pd.to_numeric(x, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]

    if len(xrange) >= 2 and len(yrange) >= 2:
        counts, xedges, yedges = np.histogram2d(x_clean, y_clean, bins=bins,range=[xrange, yrange])
    else:
        counts, xedges, yedges = np.histogram2d(x_clean, y_clean, bins=bins) 

    return counts, xedges, yedges

def slice_y_hist_by_xbin(
    counts: np.ndarray,
    yedges: np.ndarray,
    xbin_index: int,
    *,
    normalize: bool = False,
) -> Dict[str, np.ndarray]:
    """
    2Dヒストの結果から、指定 x ビンに対応する y の一次元ヒストを取り出す。

    Parameters
    ----------
    counts : shape = (nx, ny)  # np.histogram2d の戻り
    yedges : shape = (ny+1,)
    xbin_index : 0..nx-1 のビン番号
    normalize : True のとき面積1に正規化

    Returns
    -------
    dict: {
      "y_counts": (ny,),      # この x ビンの y 分布のカウント
      "y_edges":  (ny+1,),    # y ビン境界
      "y_centers":(ny,),      # y ビン中心
      "xbin_index": int
    }
    """
    nx, ny = counts.shape
    if not (0 <= xbin_index < nx):
        raise IndexError(f"xbin_index out of range: 0..{nx-1}")

    y_counts = counts[xbin_index, :].astype(float)  # 注意：counts は (nx, ny)

    if normalize and y_counts.sum() > 0:
        y_counts = y_counts / y_counts.sum()

    y_centers = 0.5 * (yedges[:-1] + yedges[1:])
    return {
        "counts": y_counts,
        "edges": yedges,
        "centers": y_centers,
        "index": int(xbin_index),
    }

def slice_y_hist_by_xrange(
    x: np.ndarray,
    y: np.ndarray,
    *,
    xedges: np.ndarray,
    ybins: int | np.ndarray,
    x_range: Tuple[float, float],
    normalize: bool = False,
) -> Dict[str, np.ndarray]:
    """
    点群から直接“x の数値範囲”でフィルタし、y の一次元ヒストを作る版。
    （ビン番号ではなく実値レンジで指定したいとき用）

    Parameters
    ----------
    x, y : 元データ
    xedges : x のビン境界（2Dヒストと合わせるなら同じものを渡す）
    ybins : y のビン数 or エッジ配列
    x_range : (xmin, xmax) で半開区間 [xmin, xmax) を想定
    normalize : True のとき面積1に正規化
    """
    xmin, xmax = x_range
    mask = (x >= xmin) & (x < xmax)
    y_sel = y[mask]

    y_counts, yedges = np.histogram(y_sel, bins=ybins)
    if normalize and y_counts.sum() > 0:
        y_counts = y_counts / y_counts.sum()
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])
    return {
        "counts": y_counts,
        "edges": yedges,
        "centers": y_centers,
        "range": np.array([xmin, xmax]),
    }


def add_sub_plot(
        fig,
        irow:int = 1,
        icol:int = 1, 
        plot_type="1d",
        data:list = None,
        labels:list = None,
        bins=[200,200],
        logsf:str = None,
        xrange:list = None,
        yrange:list = None,
        debug:bool = False,
        legends:list = None,
        dataname:str = None,
        color:str = None,
        colormap:str = "Viridis"
):

    if data is None:
        data = []

    if labels is None:
        labels = []

    if bins is None:
        bins = [200, 200]

    if logsf is None:
        logs = [False, False, False]
    else:
        logs = []
        for i in range(3):
            val_bool = True if logsf[i] == "1" else False
            logs.append(val_bool)

    if xrange is None:
        xrange = []

    if yrange is None:
        yrange = []

    xtype = '-' if logs[0] is False else 'log'
    ytype = '-' if logs[1] is False else 'log'
 
    if plot_type == '1d':
        if len(xrange) < 2:
            fig.add_trace(
                go.Histogram(x=data[0],nbinsx=bins[0],name=dataname),
                row=irow, col=icol
            )
        else:
            fig.add_trace(
                go.Histogram(
                    x=data[0],
                    xbins=dict(
                        start=xrange[0],   
                        end=xrange[1], 
                        size=xrange[2] 
                    ),
                    name=dataname
                ),
                row=irow, col=icol
            )

    elif plot_type == '2d': 
        counts, xedges, yedges = get_np_histogram2d(data=data, bins=bins, xrange=xrange, yrange=yrange)

        if logs[2]:
            counts = np.log10(counts + 1)

        bar_title = "ln(+1)" if logs[2] else "Count" 

        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])

        heatmap = go.Heatmap(
            x=xcenters,
            y=ycenters,
            z=counts.T,
            colorscale=colormap,
            colorbar=dict(
                title=bar_title
            ),
            name=dataname
        )
        
        fig.add_trace(heatmap, row=irow, col=icol)

        rows_range, cols_range = fig._get_subplot_rows_columns()
        nrows = len(rows_range)
        ncols = len(cols_range)

        index = (irow - 1) * ncols + (icol - 1)
        base_title = fig.layout.annotations[index].text 
        fig.layout.annotations[index].text = f"{base_title}, Entries:{int(counts.sum())}"

        if debug:
            total_count = counts.sum()

            max_val = counts.max()
            max_index = np.unravel_index(np.argmax(counts), counts.shape)  # (iy, ix)
            iy, ix = max_index

            xcenters = 0.5 * (xedges[:-1] + xedges[1:])
            ycenters = 0.5 * (yedges[:-1] + yedges[1:])

            x_at_max = xcenters[ix]
            y_at_max = ycenters[iy]

            min_val = counts.min()
            min_index = np.unravel_index(np.argmin(counts), counts.shape)  # (iy, ix)
            iy, ix = min_index

            x_at_min = xcenters[ix]
            y_at_min = ycenters[iy]

            print(f"[debug] Entries {total_count}, Max value {max_val} at ({x_at_max},{y_at_max}), Min value {min_val} at ({x_at_min},{y_at_min})")
    
    elif plot_type == 'scatter':
        if color is not None:
            fig.add_trace(
                go.Scatter(
                    y = data[0],
                    mode = 'lines+markers',
                    marker = dict(size = 4, color = color),
                    line = dict(width = 1, color = color),
                    name = dataname
                ),
                row=irow, col=icol
            )
        else:
            fig.add_trace(
                go.Scatter(
                    y = data[0],
                    mode = 'lines+markers',
                    marker = dict(size = 4),
                    line = dict(width = 1),
                    name = dataname
                ),
                row = irow,
                col = icol
            )            

    elif plot_type == 'fit':
        if color is not None:
            fig.add_trace(
                go.Scatter(
                    x = data[0],
                    y = data[1],
                    mode = 'lines',
                    marker = dict(
                        size = 4,
                        color = color
                    ),
                    line = dict(
                        width = 2,
                        color = color
                    ),
                    name = dataname
                ),
                row = irow,
                col = icol,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x = data[0],
                    y = data[1],
                    mode = 'lines',
                    marker = dict(size = 4),
                    line = dict(width = 2),
                    name = dataname
                ),
                row = irow,
                col = icol,
            )

    elif plot_type == 'plot':
        if color is not None:
            fig.add_trace(
                go.Scatter(
                    x = data[0],
                    y = data[1],
                    mode = 'markers',
                    marker = dict(
                        size = 8,
                        color = color
                    ),
                    line = dict(
                        width = 1,
                        color = color
                    ),
                    name = dataname),
                row = irow, 
                col = icol
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x = data[0],
                    y = data[1],
                    mode = 'markers',
                    marker = dict(size = 8),
                    line = dict(width = 1),
                    name = dataname
                ),
                row = irow,
                col = icol
            )

    elif plot_type == 'spark-hist':
        fig.add_trace(
            go.Bar(
                x = data[0],
                y = data[1],
                name = dataname
            ),
            row = irow,
            col = icol
        )
    
    elif plot_type == 'error':
        if color is not None:
            fig.add_trace(
                go.Scatter(
                    x = data[0],
                    y = data[1],
                    mode = "markers",
                    name = dataname,
                    error_y = dict(
                        type = "data",
                        array = data[3],
                        arrayminus = data[2],
                        visible = True
                    )
                ),
                row = irow,
                col = icol
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x = data[0],
                    y = data[1],
                    mode="markers",
                    marker = dict(size = 4),
                    line = dict(width = 1),
                    name = dataname,
                    error_y = dict(
                        type="data",
                        array = data[3],
                        arrayminus = data[2],
                        visible = True
                    )
                ),
                row = irow,
                col = icol
            )      

    ### update axis info
    if len(labels) >=2:
        fig.update_xaxes(
            type = xtype,
            title_text = labels[0],
            row = irow,
            col = icol
        )

        fig.update_yaxes(
            type = ytype,
            title_text = labels[1],
            row = irow,
            col = icol
        )

    ### legend option
    if legends is not None:
        if len(legends) > 5:
            fig.update_layout(
                legend=dict(
                    x = legends[0],
                    y = legends[1],   
                    xanchor = legends[2],
                    yanchor = legends[3],
                    orientation = legends[4]  
                ),
                margin=dict(r = legends[5])               
        )

def align_colorbar(fig, thickness=20, thicknessmode="pixels"):
    for trace in fig.data:
        if isinstance(trace, go.Heatmap):
            xaxis = "xaxis" if trace.xaxis == "x" else "xaxis" + trace.xaxis[1:]
            yaxis = "yaxis" if trace.yaxis == "y" else "yaxis" + trace.yaxis[1:]

            xa = fig.layout[xaxis].domain
            ya = fig.layout[yaxis].domain

            trace.update(colorbar=dict(thickness=thickness, thicknessmode=thicknessmode, x=xa[1] + 0.01, y=(ya[0] + ya[1]) / 2, len=ya[1] - ya[0]))

def build_event_and_find_pulse(input, output, dt, debug=False):

    OUTPUT_PARQUET_PATH = output

    grouped_samples, grouped_timestamp, grouped_chip, grouped_channel = build_event_usung_dt(input, dt, debug)

    filtered_grouped_samples, filtered_grouped_timestamp, filtered_grouped_chip, filtered_grouped_channel, \
        filtered_grouped_event_id, filtered_grouped_times, filtered_grouped_counts, filtered_grouped_max_sample, \
            filtered_grouped_charge, filtered_grouped_total_charge = find_pulse(grouped_samples, grouped_timestamp, grouped_chip, grouped_channel, debug)
    
    Qtot = 0

    ########### check data
    if 1:
        print(f"event_id: {len(filtered_grouped_event_id)}")
        print(f"chip: {len(filtered_grouped_chip)}")
        print(f"channel: {len(filtered_grouped_channel)}")
        print(f"timestamp: {len(filtered_grouped_timestamp)}")
        print(f"pulse: {len(filtered_grouped_samples)}")
        print(f"max_sample_timing: {len(filtered_grouped_times)}")
        print(f"multiplicity: {len(filtered_grouped_counts)}")
        print(f"max_sample: {len(filtered_grouped_max_sample)}")
        print(f"charge: {len(filtered_grouped_charge)}")
        print(f"total_charge: {len(filtered_grouped_total_charge)}")

    ############ save data
    if 1:
        # 1. SparkSession 作成
        save_spark = SparkSession.builder.appName("SaveParquet").getOrCreate()

        # 2. 例として pandas から Spark DataFrame 作成
        pdf = pd.DataFrame({
            "event_id": filtered_grouped_event_id,
            "chip": filtered_grouped_chip, 
            "channel": filtered_grouped_channel, 
            "timestamp": filtered_grouped_timestamp, 
            "pulse": filtered_grouped_samples,
            "multiplicity": filtered_grouped_counts,
            "max_sample": filtered_grouped_max_sample,
            "charge": filtered_grouped_charge,
            "total_charge": filtered_grouped_total_charge,
            "time": filtered_grouped_times
        })

        df = save_spark.createDataFrame(pdf)

        df.write.mode("overwrite").parquet(OUTPUT_PARQUET_PATH)
        print(f"saved data to {OUTPUT_PARQUET_PATH}" )

    ############ draw figure
    if 1:
        multiplicity = []
        sampling_counts = []
        filtered_multiplicity = []
        filtered_sampling_counts = []
        filtered_max_sample = []
        filtered_charge = []
        filtered_total_charge = []

        for i in range(len(grouped_samples)):
            multiplicity.append(len(grouped_samples[i])) 

            for j in range(len(grouped_samples[i])):
                sampling_counts.append(len(grouped_samples[i][j]))

        for i in range(len(filtered_grouped_samples)):
            filtered_multiplicity.append(len(filtered_grouped_samples[i])) 

            for j in range(len(filtered_grouped_samples[i])):
                filtered_sampling_counts.append(len(filtered_grouped_samples[i][j]))
                filtered_max_sample.append(max(filtered_grouped_samples[i][j]))
                filtered_charge.append(sum(filtered_grouped_samples[i][j]))

                Qtot = Qtot + sum(filtered_grouped_samples[i][j])
            
            filtered_total_charge.append(Qtot)
            Qtot = 0

        flattened_samples_2d = get_2darray_from_3darray(filtered_grouped_samples)
        index_array_2d = get_2dindex_array(flattened_samples_2d)
        flattened_samples_1d = get_1darray_from_2darray(flattened_samples_2d)
        index_array_1d = get_1darray_from_2darray(index_array_2d)
        # flattened_maxsample_1d = get_1darray_from_2darray(filtered_max_sample)
        # flattened_charge_1d = get_1darray_from_2darray(filtered_charge)

        if debug:
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    "Multiplicity (Raw)",
                    "Sampling Counts (Raw)",
                    "Filtered Multiplicity (After Pulse Finder)",
                    "Filtered Sampling Counts(After Pulse Finder)",
                    "Pulse Charge (After Pulse Finder)",
                    "Sum of Pulse Charge in 1 Event (After Pulse Finder)"
                ),
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )

            add_sub_plot(fig,1,1,'1d',[multiplicity],['Multiplicity','Frequency'])
            add_sub_plot(fig,1,2,'1d',[sampling_counts],['Sampling Count','Frequency'])
            add_sub_plot(fig,2,1,'1d',[filtered_multiplicity],['Filtered Multiplicity','Frequency'])
            add_sub_plot(fig,2,2,'1d',[filtered_sampling_counts],['Filtered Sampling Count','Frequency'])
            add_sub_plot(fig,3,1,'1d',[filtered_charge],['Pulse Charge','Frequency'])
            add_sub_plot(fig,3,2,'1d',[filtered_total_charge],['Sum of Pulse Charge in 1 Event','Frequency'])

            fig.update_layout(
                height=800,
                width=1000,
                title_text="Distributions of Multiplicity and Sampling Counts",
                showlegend=False
            )
            fig.show()

        if debug:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(
                    "Pulse after pulse finding",
                    "Charge vs MaxSample",
                ),
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )

            df = pd.DataFrame({'sample': flattened_samples_1d, 'index': index_array_1d})
            x = df["index"].to_numpy()
            y = df["sample"].to_numpy()
            add_sub_plot(fig,1,1,'2d',[x,y],['index','sample'],[30,200])

            df = pd.DataFrame({'maxsample': filtered_max_sample, 'charge': filtered_charge})
            y = df["maxsample"].to_numpy()
            x = df["charge"].to_numpy()
            add_sub_plot(fig,1,2,'2d',[x,y],['charge','maxsample'],[200,200])
          
            fig.update_layout(
                height=600,
                width=1600,
                title_text="puslse infomation",
                showlegend=False
            )
            fig.show()

        
        if debug:
            debug_timestamp =[]
            for i in range(len(filtered_grouped_timestamp)):
                debug_timestamp.append(filtered_grouped_timestamp[i][0])
            plot_timestamps_plotly(debug_timestamp)

if __name__ == "__main__":
    toml_file_path = this_file_path  / "../../../parameters.toml"

    with open(toml_file_path, "r") as f:
        config = toml.load(f)

    fileinfo = config["fileinfo"]

    BASE = fileinfo["base_output_path"]
    FILE = fileinfo["input_file_name"]

    INPUT_PARQUET_PATH = BASE + "/" + FILE + ".parquet"
    OUTPUT_PARQUET_PATH = BASE + "/" + FILE + "_pulse.parquet"

    analysinfo = config["analysis"]
    
    MAX_DELTA = analysinfo["time_window"] 

    build_event_and_find_pulse(INPUT_PARQUET_PATH, OUTPUT_PARQUET_PATH, MAX_DELTA, True)
