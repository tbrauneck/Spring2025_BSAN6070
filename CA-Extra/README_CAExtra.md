# CA-Extra Clustering

## Description
This program uses the k-means algorithm to cluster mall customers based on their spending and demographics.<br>

## Authoring
Authored by Tina Brauneck. Initially coded in April 2025. This program was generated as an assignment in a graduate level machine learning class. <br>

## Data Source
Data was sourced from: https://github.com/ArinB/MSBA-CA-Data/raw/main/CA06/Mall_Customers.csv<br>

This dataset includes 200 observations and five attributes:
1.	CustomerID: Unique ID for each customer
2.	Gender: Male or Female
3.	Age: Age of the customer
4.	Annual Income (k$): Annual income of the customer in thousands of dollars
5.	Spending Score (1-100): A score assigned by the mall based on customer behavior and spending nature (higher scores indicate higher spending). <br>


## Instructions
All steps should be run in order. Important notes:
1. There is a known memory leak in sci-kit learn's k-means package. It is critical that you run the step that sets the number of threads to 1 before running the k-means model. Follow the comments in the file. Warnings about this leak may be ignored as long as this step has been run.
2. All findings noted in the markdown are based on a random seed of 300, sci-kit learn version 1.6.1 and the use of k-mean++ for initialization. Cluster results can vary greatly depending on initialization, so these conditions must be met to ensure results consistent with the markdown.
3. Github cannot preview the plotly express 3-D scatterplot. To view or interact with this plot, it is recommended that you run the code in JupyterLab.
<br>

## Versioning
This code was generated on JupyterLab version 4.2.5, using sci-kit learn version 1.6.1. Update the code as needed for use in later versions.

## Packages
The following packages were used. These must be installed prior to running the code if not already installed. <br>

os<br>
matplotlib<br>
pandas<br>
numpy<br>
seaborn<br>
plotly<br>
plotnine<br>
sklearn<br>

