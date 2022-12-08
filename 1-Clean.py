# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 10:43:36 2022

@author: RPopocovski
"""

import pandas as pd
import functions

'''
Data Sets
The data is contained in three files:

portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
profile.json - demographic data for each customer
transcript.json - records for transactions, offers received, offers viewed, and offers completed
Here is the schema and explanation of each variable in the files:

portfolio.json

id (string) - offer id
offer_type (string) - type of offer ie BOGO, discount, informational
difficulty (int) - minimum required spend to complete an offer
reward (int) - reward given for completing an offer
duration (int) - time for offer to be open, in days
channels (list of strings)

profile.json

age (int) - age of the customer
became_member_on (int) - date when customer created an app account
gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
id (str) - customer id
income (float) - customer's income

transcript.json

event (str) - record description (ie transaction, offer received, offer viewed, etc.)
person (str) - customer id
time (int) - time in hours since start of test. The data begins at time t=0
value - (dict of strings) - either an offer id or transaction amount depending on the record
'''


# read in the json files
portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('data/profile.json', orient='records', lines=True)
transcript = pd.read_json('data/transcript.json', orient='records', lines=True)

pd.set_option('display.max_columns', None) # setting pandas to display all dataframe columns

# Exploration of individaul datasets
print('Exploring the individual datasets....')

print('\n Percentage of NaN values in *transcript* dataset by columns:')
print (functions.nan_percentage(transcript))

print('\n Percentage of NaN values in *profile* dataset by columns:')
print (functions.nan_percentage(profile))# columns gender and income both contain 14.6% of NaN values

'''could the lack of gender and income information of a specific user be a an information on itself? '''
profile['gender'].fillna('ND', inplace=True) # replacing NaN values of gender with ND as Not Discolsed

print('Out of %d NaN values in \'income\' column, %d entires also don\'t have discolsed gender.'%(profile.income.isna().sum(),profile[profile['gender']=='ND'].income.isna().sum() ) )
 # all profiles with missing income also don't have disclosed gender
 
median_income = profile['income'].median()
profile['income'].fillna(value=median_income, inplace=True)
print ('Missing gender values have been encoded as \'ND\' (Not Disclosed) whereas the missing income has been filled out with the median income. ')


# Transformation
print('\n\nTransforming the individual datasets....\n')
user_no  = transcript['person'].unique().size # number of unique users

transcript_full = pd.concat([transcript.drop(['value'], axis=1), transcript['value'].apply(pd.Series)], axis=1) # expanding the dictionaries in value column to seperate columns

transcript_full[~transcript_full['offer_id'].isna()]['offer id'].unique() #columns 'offer_id' and 'offer id' contain the same information
transcript_full[~transcript_full['offer id'].isna()]['offer_id'].unique()

transcript_full['offer_id'].fillna(transcript_full['offer id'], inplace=True) # combining two offer id columns into one
transcript_full.drop('offer id', axis =1, inplace=True)
#transcript_full['offer_id'].unique().size

print (transcript_full.time.unique()) #timestamps are in the range of 0-719 with interval of 6 and represent time period (in hours) from the start of the 30-day experiment
print ('Timestamps in event log cover the range of 0-719 (in increments of 6) and represent time period (in hours) from the start of the 30-day experiment.\n')


# =============================================================================
# transcript_byuser = transcript_full.groupby(['person','offer_id'])#event log grouped by user name and event type
# transcript_byuser_count = transcript_full.groupby(['person','offer_id','event']).size().reset_index(name='number_of_events')#['time', 'event']#.size().reset_index(name='number_of_events') # logged events grouped by user name and event type
# transcript_byuser_count.sort_values(['person','offer_id','event','number_of_events'], ascending=False, inplace=True)
# transcript_byuser_count # the same offer can be received by the same user multiple times
# =============================================================================

# using one group (one person and one offer id) to explore the needed transformations before applying it to the entire dataframe 
# =============================================================================
# for index, group in transcript_byuser:
#     if group[group['event']=='offer completed'].shape[0]>1:
#         group_example = group
#         break
#     
# group_example.drop(['amount', 'reward'], axis=1, inplace=True)
# print (group_example)
# 
# group_example_pivoted= pd.pivot_table(group_example, index=['person', 'offer_id'], columns = ['event'], values = 'time', aggfunc=functions.agg_list, fill_value=0) # pivoting event type into columns where the values are a list of timestamps
# group_example_pivoted.reset_index(inplace=True)
# 
# group_example_pivoted= group_example_pivoted.merge(portfolio, left_on='offer_id', right_on = 'id') #merging event log with offer details from protfolio dataframe
# group_example_pivoted.drop(['id'], axis=1, inplace=True) #removing duplicated column
# 
# group_example_pivoted['duration hours']=group_example_pivoted['duration']*24
# group_example_pivoted['received count'] = group_example_pivoted['offer received'].str.len()
# group_example_pivoted['completed count'] = group_example_pivoted['offer completed'].str.len()
# 
# #new columns with count of event sequnces
# group_example_pivoted['r-v'] = group_example_pivoted.apply(functions.interaction_counter, args=('offer received', 'offer viewed'), axis=1)
# group_example_pivoted['r-v-c'] = group_example_pivoted.apply(functions.influenced_counter, axis=1)
# #new columns with offer success metrics (impact - % of received offers that were viewed and then completed, misplaced impact - % of received offers that were completed without being viewed first)
# group_example_pivoted['impact'] = group_example_pivoted['r-v-c']/group_example_pivoted['received count']
# group_example_pivoted['misplaced impact']=(group_example_pivoted['completed count'] - group_example_pivoted['r-v-c'])/group_example_pivoted['received count']
# 
# print(group_example_pivoted)
# group_example_pivoted.to_csv('example.csv')
# 
# =============================================================================
transcript_sliced=transcript_full#.drop(['amount', 'reward'], axis=1)
transcript_pivoted= pd.pivot_table(transcript_sliced, index=['person', 'offer_id'], columns = ['event'], values = 'time', aggfunc=functions.agg_list, fill_value=0) # pivoting event type into columns where the values are a list of timestamps
transcript_pivoted.reset_index(inplace=True)

transcript_pivoted= transcript_pivoted.merge(portfolio, left_on='offer_id', right_on = 'id') #merging event log with offer details from protfolio dataframe
transcript_pivoted.drop(['id'], axis=1, inplace=True) #removing duplicated column

transcript_pivoted['duration hours']=transcript_pivoted['duration']*24
transcript_pivoted['received count'] = transcript_pivoted['offer received'].str.len()
transcript_pivoted['completed count'] = transcript_pivoted['offer completed'].str.len()
transcript_pivoted['completed count'].fillna(0, inplace=True) #filling Nan values with 0
transcript_pivoted['completed count'] = transcript_pivoted['completed count'].astype(int)

#new columns with count of event sequnces
transcript_pivoted['r-v'] = transcript_pivoted.apply(functions.interaction_counter, args=('offer received', 'offer viewed'), axis=1)
transcript_pivoted['r-v-c'] = transcript_pivoted.apply(functions.influenced_counter, axis=1)
#new columns with offer success metrics (impact - % of received offers that were viewed and then completed, misplaced impact - % of received offers that were completed without being viewed first)
transcript_pivoted['impact'] = transcript_pivoted['r-v-c']/transcript_pivoted['received count']
transcript_pivoted['misplaced impact']=(transcript_pivoted['completed count'] - transcript_pivoted['r-v-c'])/transcript_pivoted['received count']


#transcript_pivoted =  transcript_pivoted.merge(profile, left_on = 'person', right_on='id') #adding demographics data to the dataframe
#transcript_pivoted.drop('id', axis=1, inplace=True)

print(transcript_pivoted.head(5))
print('Event log data and offer details data have been used to create a unique user-offer-demo DataFrame that contains details of events, counts of event type sequences, offer success metrics and main demographic characteristics. (Sample shown above).')

transcript_pivoted.to_csv('dataset_cleaned.csv')
print('\n Cleaned transcript has been saved to a seperate csv file.') 
