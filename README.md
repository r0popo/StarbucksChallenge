# StarbucksChallenge
Identifying customer groups that are most responsive to different offer types. Used as a capstone project for the Udacity's Data Scientist Nanodegree.

### Table of Contents

1. [Dependencies](#dependencies)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)
6. [References](#references)

## Dependencies <a name="dependencies"></a>
This project was developed in Python 3.10+.\
Main packages used are:
+ [pandas](https://pandas.pydata.org/docs/)
+ [seaborn](https://seaborn.pydata.org/index.html) >=0.12
+ [matplotlib](https://matplotlib.org/stable/index.html)
+ [joypy](https://github.com/leotac/joypy)
+ [scipy](https://docs.scipy.org/doc/scipy/) 

A full list of requirements can be found in the 'requirements.txt' file which can be used to create conda virtual environment with:
```
conda create --name <env> --file requirements.txt
```

## Project Motivation<a name="motivation"></a>


## File Descriptions<a name="files"></a>
+ data.zip - original datasets provided for the purposes of StarbucksChallenge. The folder contains three .csv files and a .txt file with description of each datasets
+ functions.py - python script with all custom functions used throughout the project
+ 1-Clean.py - python script used for initial dataset exploration and data wrangling
+ 2-Analyse.py - python script used for data visualisation and analysis
+ dataset_cleaned.csv - csv file exported after running 1-Clean step
+ plots.zip - all plots generated during 2-Analyse step
+ KS_test.csv and KS_test_offercomp.csv - csv files generated during 2-Analyse step that contain results of the Kolmogorov–Smirnov test


## Results<a name="results"></a>
The main findings of the code can be found at the post available [here](https://medium.com/@ropopo/).

## Licensing, Authors, Acknowledgements<a name="licensing"></a>


This project is inspired and submitted for the purposes of Udacity's Data Science Nanodegree. You can find additional information about the course [here](https://udacity.com/course/data-scientist-nanodegree--nd025).

## References <a name="references"></a>
1. https://github.com/leotac/joypy
2. https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
3. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html
4. https://seaborn.pydata.org/generated/seaborn.FacetGrid.html
5. [Visualization with Seaborn](https://jakevdp.github.io/PythonDataScienceHandbook/04.14-visualization-with-seaborn.html)
6. [Seaborn Styling, Part 1: Figure Style and Scale](https://www.codecademy.com/article/seaborn-design-i)
7. [A Gentle Introduction to Probability Density Estimation](https://machinelearningmastery.com/probability-density-estimation/)
8. [How to Compare Two or More Distributions](https://towardsdatascience.com/how-to-compare-two-or-more-distributions-9b06ee4d30bf)
9. [Customer Demographics & Segmentation Analysis with Simple Python](https://medium.com/analytics-vidhya/customer-demographics-segmentation-analysis-with-simple-python-cdd2e6d35f2e)
