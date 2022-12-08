# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 11:39:12 2022

@author: RPopocovski
"""

import pandas as pd
import numpy as np

def nan_percentage (df):
    '''INPUT:
        df - (pandas DataFrame)  dataframe for column analysis
        
        OUTPUT:
        nan_percentage - (list) contaians column name as a key and calculated percentaage of NaN values in that column
        '''
    columns = df.columns.tolist()
    nan_percentage = dict()
    for col in columns:
        nan_calc = df[col].isna().sum()/df[col].count()
        nan_percentage[col]=nan_calc
        
    return  nan_percentage

def agg_list(values):
    '''
    INPUT:
        values - (int) values to be aggregated   
    OUTPUT:
        values - (list) list of values
    '''
            
    return list(values)


def interaction_counter(row, column1, column2 ):
    '''
    INPUT:
        row - (pandas Series) one row of the dataframe
        column1 - (str) name of the column
        column2- (str) name of the column we want to compare with column1
        
    OUTPUT:
        count - (int) number of offers that were viewed in the time period of offer duration
    '''
    if row.loc[column1] == 0 or row.loc[column2] == 0:
        return 0
    
    count = 0
    for i in row.loc[column1]:
        for j in row.loc[column2]:
            if j>=i and j<=(i+row.loc['duration hours']):
                count+=1
                break
            
    return count

def influenced_counter(row):
    '''
    INPUT:
        row - (pandas Series) one row of the dataframe
    OUTPUT:
        count - (int) number of offers that were first viewed and then completed in the time period of offer duration
    '''
    if row.loc['offer completed'] == 0 or row.loc['offer viewed'] == 0:
        return 0
    count = 0
    
    for c in row.loc['offer completed']:
        found=False
        for r in row.loc['offer received']:
            if c>=r and c<=(r+row.loc['duration hours']):
                for v in row.loc['offer viewed']:
                    if v>=r and v<=c:
                        count+=1
                        found=True
                        break
                if found:
                    break
                
    return count


def binning(row, column):
    '''
    INPUT:
        row - (pandas Series) one row of the dataframe
        column - (str) name of the column
    OUTPUT:
        interval - (str) name of the interval the value falls in
    '''
    bins = np.array([0.0, 0.25, 0.5, 0.75, 1])
    interval = ''
    i = row.loc[column]
    if np.digitize(i, bins, right=False)==1:
        interval = '0.00 - 0.25'
    elif np.digitize(i, bins, right=False)==2:
        interval = '0.25 - 0.50'
    elif np.digitize(i, bins, right=False)==3:
        interval = '0.50 - 0.75'
    else:
        interval = '0.75 - 1.00'
    return interval