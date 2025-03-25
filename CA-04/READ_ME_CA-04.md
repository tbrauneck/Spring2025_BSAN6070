# CA-04 Ensemble Models

## Description
This program uses ensemble classifier models to predict income class (>$50K or <$50K) based on 8 features:<br>
<br>
• Hours Worked per Week<br>
• Occupation Category<br>
• Marriage Status & Relationships<br>
• Capital Gain<br>
• Race-Sex Group<br>
• Number of Years of Education<br>
• Work Class<br>
• Age<br>

## Authoring
Authored by Tina Brauneck. Initially coded in March 2025.

## Data Source
The dataset was obtained from the Census Bureau and represents people's salaries, along with nine demographic variables. The following is a description of our dataset:<br>
• Number of target classes: 2 ('>50K' and '<=50K') [ Labels: 1, 0 ]<br>
• Number of attributes (Columns): 11<br>
• Number of instances (Rows): 48,842<br>

Access the dataset here: https://github.com/ArinB/MSBA-CA-03-Decision-Trees/blob/master/census_data.csv?raw=true<br>

All data is binned in the source file. Testing and training observations have been flagged in the source data using the 'flag' column.<br>

## Instructions
All steps should be run in order. Note that performance variables are not renamed in each step, so if run out of order, the metics quoted may apply to a different model than intended. Take caution when selectively running blocks of code.
Predictions can be made by altering the "user input" dictionary defined in the "Prediction" section. Values must match verbatim.


## Versioning
This code was generated on JupyterLab version 4.2.5. The code may need to be updated for use in later versions.

## Packages
The following packages were used. These must be installed prior to running the code if not already installed. <br>

matplotlib<br>
pandas<br>
numpy<br>
seaborn<br>
time<br>
plotnine<br>
sklearn<br>
scipy.stats<br>
graphviz<br>

