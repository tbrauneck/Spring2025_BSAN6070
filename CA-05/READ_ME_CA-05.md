# CA-05 KNN Movie Recommender 

## Description
This program is a simple movie recommender model that uses the KNN algorithm to choose the most closely related movie based on genre. Preceding the model is a data quality report and a brief exploration of the a dataset.
<br>


## Authoring
Authored by Tina Brauneck. Initially coded in April 2025 as an assignment for a graduate level machine learning class.

## Data Source
The dataset is a small sample of IMDb data from the UCI Machine Learning Repository. It contains only 30 observations. Key features and identifiers in the dataset are movie name, IMDb, and genre. <br>
Access the dataset here: https://github.com/ArinB/MSBA-CA-Data/raw/main/CA05/movies_recommendation_data.csv <br>


## Instructions
All steps can be run in order. A sample prediction for a historical, biographical drama called "The Post" has been queued up in the "Recommendations" section. Predictions for other movies can be made by altering the "watched_movie_data" dictionary defined in the "Recommendations" section. A '1' indicates an association with the genre. A '0' indicates no association. Movies may have multiple dramas.


## Versioning
This code was generated on JupyterLab version 4.2.5. The code may need to be updated for use in later versions.


## Required Packages
The following packages were used. These must be installed prior to running the code if not already installed. <br>

math<br>
matplotlib<br>
numpy<br>
pandas<br>
plotnine<br>
sklearn<br>
