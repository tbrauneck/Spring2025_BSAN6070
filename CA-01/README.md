READ ME

# CA-01 Housing Predictions

## Versioning:

This code was generated on JupyterLab version 4.2.5. Note that some of the code may generate deprecation warnings. It will still run effectively on this version, however, the code may need to be updated for use in later versions.

## Packages:

The following packages were used. These must be installed prior to running the code if not already installed. An optional install code block has been provided with some of the less common packages.

matplotlib <br>
pandas <br>
numpy <br>
seaborn <br>
plotly <br>
plotnine <br>

## Context:

This program was generated for a graduate-level business analytics course. It demonstrates concepts related to data quality assessment, data exploration, and pre-processing.

## Data Source:

The data used in this program is house pricing data for 1,460 homes. The source file and relevant dictionary are available publicly on GitHub.
	Dictionary: https://github.com/ArinB/MSBA-CA-Data/raw/refs/heads/main/CA01/data_description.txt
	Source file: https://github.com/ArinB/MSBA-CA-Data/raw/refs/heads/main/CA01/house-price-train.csv

Most sections of this program have been designed in reusable blocks that can fit any csv source file. However, some of the pre-processing steps are source-specific; individual feature names or values are called. These lines of code will need modification or removal if a new source is used. Sizing may also require modification so that there are the correct number of plots and figures fit a standard screen width.  

## Running the Code:

The code includes detailed markdown with a report-like structure. A table of contents has been provided at the beginning.

All lines should be run in order. Aside from installations, no modification should be required. Note that plot matrices may take several minutes or more to run, depending on the speed of your machine. Individual plots in the initial scatter plot matrix may be too small to read. Recommended options are:

1) Save the matrix and view in a program that allows zoom
2) An abbreviated matrix with only continuous variables of moderate-high cardinality has been provided. This may be easier to view.
3) An interactive scatterplot matrix has been provided. Hover the cursor over a plot to see the variable names; this shows which features are compared.


