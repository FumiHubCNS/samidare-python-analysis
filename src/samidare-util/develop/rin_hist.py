"""!
@file rin_hist.py
@version 1
@author FumiHubCNS
@date 2025-08-20T09:35:34+09:00
@brief template text
"""
import click
import pathlib
import datetime
import matplotlib.pyplot as plt
import numpy as np

from pyspark.sql import DataFrame
from pyspark.sql.functions import explode, posexplode, row_number, lit
from pyspark.ml.feature import Bucketizer
from pyspark.sql.window import Window


this_file_path = pathlib.Path(__file__).parent

def Hist1D(dataFrame: DataFrame, colName: str, nbins: int, range: tuple[float, float]) -> plt:
    """
    Plot 1D histogram of the column named "colName". Assuming the column stores a value per row.
    Uses spark to count bin contents. pandasHist1d module could be faster for a short DataFrame

    Parameters
    ----------
    dataFrame: Input DataFrame that contains a column named colName
    colName: Column name to plot
    nbis: Number of bins
    range: Histogram range as [min, max]

    Returns
    -------
    1D histogram as matplotlib.pyplot.plt
    """
    # Define bin edges (n bins between -range[0] and range[1])
    bin_edges = np.linspace(range[0], range[1], num=nbins).tolist()

    # Add underflow and overflow bins to an extended list
    ex_bin_edges = bin_edges.copy()
    ex_bin_edges.append(np.inf)
    ex_bin_edges.insert(0, -np.inf)

    # Create a Bucketizer with the extended list of bins
    bucketizer = Bucketizer(splits=ex_bin_edges, inputCol=colName, outputCol="bin")

    # Apply the Bucketizer to the DataFrame
    binned_df = bucketizer.transform(dataFrame.select(colName))

    # Group by bin and count the occurrences in each bin and remove Null bins
    histogram_df = binned_df.groupBy("bin").count().orderBy("bin")
    histogram_df = histogram_df.filter(histogram_df["bin"].isNotNull())

    # Collect the histogram data from Spark to local
    histogram_data = histogram_df.collect()

    # Calculate bin centers
    bin_edges = np.linspace(range[0], range[1], num=nbins)
    counts = np.zeros(len(bin_edges)-1)

    # Populate the counts array based on the histogram data
    # Also, counts statistics for under/overflowed bins
    underflow = 0
    overflow = 0
    inrange = 0
    for row in histogram_data:
        bin_index = int(row['bin'])  # Get the bin index
        if bin_index == 0: # The underflow bin
            underflow = row['count']
        if bin_index == nbins: # The overflow bin
            overflow = row['count']
        else:
            counts[bin_index-1] = row['count']  # Populate the count for the bin
            inrange = inrange + row['count']

    # Calculate bin centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_centers[1] - bin_centers[0]

    # Plot the histogram
    plt.bar(bin_centers, counts, width=bin_width)

    # Add labels and title
    plt.xlabel(colName)
    plt.ylabel("Frequency")
    plt.title("Hist1D " + colName + " values")

    # Print statistics
    print("Total entries: {}, Underflow: {}, Inside: {}, Overflow: {}".format(inrange+underflow+overflow, underflow, inrange, overflow))

    return plt

def Hist1DArrays(dataFrame: DataFrame, colName: str, nbins: int, range: tuple[float, float]) -> plt:
    """
    Plot 1D histogram of the column named colName.
    The column stores an array of values.

    Parameters
    ----------
    dataFrame: Input DataFrame
    colName: Column name to plot
    nbis: Number of bins
    range: Histogram range as [min, max]

    Returns
    -------
    1D histogram as matplotlib.pyplot.plt
    """
    dataFrame = dataFrame.select(colName)
    exploded_df = dataFrame.select(explode(colName).alias(colName))

    return Hist1D(exploded_df, colName, nbins, range)

def Hist2D(dataFrame: DataFrame, colName: tuple[str, str], nbins: tuple[int, int], range: tuple[tuple[float, float], tuple[float, float]]) -> plt:
    """
    Plot 2D histogram of the column named colName["x", "y"]. Assuming the column stores a value per row

    Parameters
    ----------
    dataFrame: Input DataFrame
    colName: Column names to plot [x, y]
    nbis: Number of bins [nbinsx, nbinsy]
    range: Histogram range as [x[min, max], y[min, max]]

    Returns
    -------
    2D histogram as matplotlib.pyplot.plt
    """
    # Define bin edges
    bin_edges_x = np.linspace(range[0][0], range[0][1], num=nbins[0]).tolist()
    bin_edges_y = np.linspace(range[1][0], range[1][1], num=nbins[1]).tolist()
    ex_bin_edges_x = bin_edges_x.copy()
    ex_bin_edges_y = bin_edges_y.copy()
    ex_bin_edges_x.append(np.inf)
    ex_bin_edges_y.append(np.inf)
    ex_bin_edges_x.insert(0,-np.inf)
    ex_bin_edges_y.insert(0,-np.inf)

    # Create a Bucketizer
    bucketizer_x = Bucketizer(splits=ex_bin_edges_x, inputCol=colName[0], outputCol="bin_x")
    bucketizer_y = Bucketizer(splits=ex_bin_edges_y, inputCol=colName[1], outputCol="bin_y")

    # Apply the Bucketizer to the DataFrame
    binned_df = bucketizer_x.transform(dataFrame.select(colName[0], colName[1]))
    binned_df = bucketizer_y.transform(binned_df)

    # Group by bin and count the occurrences in each bin and remove Null bins
    binned_df = binned_df.filter(binned_df["bin_x"].isNotNull() & binned_df["bin_y"].isNotNull())
    histogram_2d_df = binned_df.groupBy("bin_x","bin_y").count()

    # Collect the histogram data from Spark to local
    histogram_data = histogram_2d_df.collect()

    # Extract bin indices and counts from the collected data
    counts = np.zeros((len(bin_edges_x) - 1, len(bin_edges_y) - 1))

    # Populate the 2D counts array
    # statArray is an 3 x 3 numpy.ndarray with underflow, inrange, overflow statistics for x and y
    statArray = np.zeros(shape=(3,3))
    for row in histogram_data:
        bin_x = int(row['bin_x'])
        bin_y = int(row['bin_y'])
        idx = 1
        idy = 1
        if bin_x == 0:
            idx = 0
        elif bin_x == nbins[0]:
            idx = 2
        if bin_y == 0:
            idy = 0
        elif bin_y == nbins[1]:
            idy = 2
        if idx == 1 and idy == 1:
            counts[bin_x-1, bin_y-1] = row['count']
        statArray[idx,idy] = statArray[idx,idy] + row['count']

    # Plot the 2D histogram using a heatmap
    plt.imshow(counts.T, extent=[range[0][0], range[0][1], range[1][0], range[1][1]], origin='lower', aspect='auto', cmap='viridis')

    # Add labels and title
    plt.xlabel(colName[0])
    plt.ylabel(colName[1])
    plt.title("2D Histogram of " + colName[1] + " vs " + colName[0])

    # Show the color bar
    plt.colorbar(label='Counts')

    # Print statistics
    print("Statistics:")
    print(statArray)
    return plt

def Hist2DArrays(dataFrame: DataFrame, colName: tuple[str, str], nbins: tuple[int, int], range: tuple[tuple[float, float], tuple[float, float]]) -> plt:
    """
    Plot 2D histogram of the column named colName["x", "y"]. Assuming the column stores an array that is aligned with each other

    Parameters
    ----------
    dataFrame: Input DataFrame
    colName: Column names to plot [x, y]
    nbis: Number of bins [nbinsx, nbinsy]
    range: Histogram range as [x[min, max], y[min, max]]

    Returns
    -------
    2D histogram as matplotlib.pyplot.plt
    """
    dataFrame = dataFrame.select(colName[0],colName[1])
    windowSpec = Window.orderBy(lit(0))
    dataFrame = dataFrame.withColumn("row", row_number().over(windowSpec))
    exploded_pos_x_df = dataFrame.select(posexplode(colName[0]).alias("idx", colName[0]),"row")
    exploded_pos_y_df = dataFrame.select(posexplode(colName[1]).alias("idx", colName[1]),"row")
    exploded_df = exploded_pos_x_df.join(exploded_pos_y_df, ["row", "idx"])

    return Hist2D(exploded_df, colName, nbins, range)