# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 12:12:23 2022

@author: RPopocovski
"""

import pandas as pd
from datetime import datetime
from matplotlib.dates import date2num
import functions
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from IPython.display import display
import seaborn as sns
from joypy import joyplot

#Setting the style for seaborn
sns.set() # set the style with Seaborn instead of Matplotlib
sns.set_style("whitegrid", {'axes.grid' : False})
sns.set_style("ticks")
sns.color_palette('Set2')
sns.despine()

#Setting pandas to display all dataframe columns
pd.set_option('display.max_columns', 50) 

#Reading in needed data
profile = pd.read_json('data/profile.json', orient='records', lines=True)
#portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
user_offer = pd.read_csv('dataset_cleaned.csv', index_col=0)

#Converting a column into datetime
profile['became_member_on'] = profile['became_member_on'] .astype('str')
profile['became_member_on']= pd.to_datetime(profile['became_member_on'].apply(lambda x: datetime.strptime(x, '%Y%m%d')))


#Grouping the events by user and offer type in odred to focus the following 
#analysis on different offer types regardless of the channel of distribution
user_offer = user_offer.groupby(['person','offer_type']).mean(numeric_only=True)\
    .reset_index()
#user_offer.head(5)


#Merging user-offer dataset with the customer demographics

#adding demographics data to the dataframe
user_offer =  user_offer.merge(profile, left_on = 'person', right_on='id') 
user_offer.drop('id', axis=1, inplace=True)

#Slicing the dataset to ignore the events made by individuals for whom we 
#don't know age as we can't use them for defining demographic groups
uod_filtered = user_offer[user_offer['age']<115] 
#uod_filtered.head(5)
profile_filtered = profile[profile['age']<115]


#Plotting customer demographics data
gender = sns.countplot(profile_filtered, x='gender', palette='flare')
gender.set(title = 'Distribution of Gender', xlabel ='Gender', ylabel = 'Count')
sns.despine()
plt.savefig('plots/gender_distribution.png', pad_inches= 0.15, dpi = 450)
plt.show()

age = sns.histplot(profile_filtered, x='age', binrange=(15,100),  binwidth=(5),\
                   color = (0.91262605, 0.52893336, 0.40749715), kde = True)
age.set(title = 'Distribution of Age', xlabel ='Age')
sns.despine()
plt.savefig('plots/age_distribution.png', pad_inches= 0.15, dpi = 450)
plt.show()

income = sns.histplot(profile_filtered, x='income',\
                  binrange=(20000,120000),binwidth=(5000),\
                  color = (0.91262605, 0.52893336, 0.40749715), kde = True)
income.set(title = 'Distribution of Income', xlabel ='Income ($)')
sns.despine()
plt.savefig('plots/income_distribution.png', pad_inches= 0.15, dpi = 450)
plt.show()


mem_bins = pd.date_range(start='2013-06-01',end='2019-01-01',freq='3M')
mem = sns.histplot(profile_filtered, x='became_member_on', bins=date2num(mem_bins),\
                   color = (0.91262605, 0.52893336, 0.40749715), kde = True)
mem.set(title = 'Distribution of Membership Start', xlabel ='Year')
sns.despine()
plt.savefig('plots/membership_distribution.png', pad_inches= 0.15, dpi = 450)
plt.show()


age_dist = joyplot(profile_filtered, by='gender', column='age',\
                   colormap=sns.color_palette("flare", as_cmap=True))
plt.xlabel('Age')
plt.title("Age Distribution by Gender")
plt.savefig('plots/age_distribution_bygender.png', pad_inches= 0.15, dpi = 450)
plt.show()


income_dist = joyplot(profile_filtered, by='gender', column='income',\
                      colormap=sns.color_palette("flare", as_cmap=True))
plt.xlabel('Income ($)')
plt.title("Income Distribution by Gender")
ax = plt.gca()
ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: '{:,.0f}'.format(x/1000) + 'K'))
plt.savefig('plots/income_distribution_bygender.png', pad_inches= 0.15, dpi = 450)
plt.show()



#Defining new columns with 4 equidistant intervals for simplification of further analysis
uod_filtered['Impact interval'] = uod_filtered.apply(functions.binning,\
                                                     args=('impact',), axis=1)
uod_filtered['Misplaced impact interval'] = uod_filtered.apply(functions.binning,\
                                            args=('misplaced impact',), axis=1)


# defining the order of the intervals
impact_hue_order = ['0.00 - 0.25', '0.25 - 0.50', '0.50 - 0.75', '0.75 - 1.00'] 

#Plotting graphs for Impact across different ages
age_count = sns.FacetGrid(uod_filtered, col='gender',row='offer_type',\
                          col_order=['F', 'M', 'O'], margin_titles=True,\
                          sharex=False, sharey=True, gridspec_kws={"hspace":0.15},\
                          legend_out=True, hue='Impact interval',\
                          hue_order = impact_hue_order, palette='flare')
age_count = age_count.map_dataframe(sns.histplot,  x='age', common_norm = True,\
                          binrange=(15,100),  binwidth=(5), kde=True)
age_count.set(xlabel ='Age', ylabel = 'Count')
age_count.fig.subplots_adjust(top=0.9) # adjust the top margin
age_count.fig.suptitle('Impact by Age')
age_count.add_legend()
sns.move_legend(age_count, "upper left", bbox_to_anchor=(0.87, 0.935),\
                title=None, frameon=False)
leg = age_count._legend
# =============================================================================
# for t in leg.texts:
#     # truncate label text to 4 characters
#     t.set_text(t.get_text()[:4])
# =============================================================================
leg.set_title('Impact')
plt.savefig('plots/impact_byage_count.png', pad_inches= 0.15, dpi = 450)


age_perc = sns.FacetGrid(uod_filtered[uod_filtered['offer_type']!='informational'],\
                    col='gender', col_order=['F', 'M', 'O'], row='offer_type',\
                    margin_titles=True, sharex=False, sharey=True,\
                    gridspec_kws={"hspace":0.15}, legend_out=True)
age_perc = age_perc.map_dataframe(sns.histplot, stat ='percent',  x='age',\
                    common_norm = True,  binrange=(15,100),  binwidth=(5),\
                    kde=True, hue='Impact interval', hue_order = impact_hue_order,\
                    palette='flare')
age_perc.set(xlabel ='Age', ylabel = 'Percentage %')
age_perc.fig.suptitle('Impact by Age')
plt.savefig('plots/impact_byage_perc.png', pad_inches= 0.15, dpi = 450)


#Plotting graphs for Misplaced Impact across different ages
age_count2 = sns.FacetGrid(uod_filtered, col='gender',row='offer_type',\
                        col_order=['F', 'M', 'O'], margin_titles=True,\
                        sharex=False, sharey=True, gridspec_kws={"hspace":0.15},\
                        legend_out=True, hue='Misplaced impact interval',\
                        hue_order = impact_hue_order, palette='flare')
age_count2 = age_count2.map_dataframe(sns.histplot,  x='age', common_norm = True,\
                        binrange=(15,100),  binwidth=(5), kde=True)
age_count2.set(xlabel ='Age', ylabel = 'Count')
#sns.color_palette("rocket_r", as_cmap=True)
age_count2.fig.subplots_adjust(top=0.9) # adjust the Figure in rp
age_count2.fig.suptitle('Misplaced Impact by Age')
age_count2.add_legend()
sns.move_legend(age_count2, "upper left", bbox_to_anchor=(0.815, 0.935),\
                title=None, frameon=False)
leg = age_count2._legend
leg.set_title('Misplaced Impact')
plt.savefig('plots/misimpact_byage_count.png', pad_inches= 0.15, dpi = 450)


age_perc2 = sns.FacetGrid(uod_filtered[uod_filtered['offer_type']!='informational'],\
                    col='gender',col_order=['F', 'M', 'O'], row='offer_type',\
                    margin_titles=True, sharex=False, sharey=True,\
                    gridspec_kws={"hspace":0.15})
age_perc2 = age_perc2.map_dataframe(sns.histplot, stat ='percent',  x='age',\
                    common_norm = True,  binrange=(15,100),  binwidth=(5),\
                    kde=True, hue='Misplaced impact interval',\
                    hue_order = impact_hue_order, palette='flare')
age_perc2.set(xlabel ='Age', ylabel = 'Percentage %')
age_perc2.fig.suptitle('Misplaced Impact by Age')
plt.savefig('plots/misimpact_byage_perc.png', pad_inches= 0.15, dpi = 450)



#Plotting graphs for Impact across different income groups
income_count = sns.FacetGrid(uod_filtered, col='gender',row='offer_type',\
                    col_order=['F', 'M', 'O'], margin_titles=True,\
                    sharex=False, sharey=True, gridspec_kws={"hspace":0.15},\
                    legend_out=True, hue='Impact interval',\
                    hue_order = impact_hue_order, palette='flare')
income_count = income_count.map_dataframe(sns.histplot,  x='income',\
                    common_norm = True,  binrange=(20000,120000),\
                    binwidth=(5000), kde=True)
income_count.set(xlabel ='Income ($)', ylabel = 'Count')
for ax in income_count.axes.flat:
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: '{:,.0f}'.format(x/1000) + 'K'))
income_count.fig.subplots_adjust(top=0.9)
income_count.fig.suptitle('Impact by Income')
income_count.add_legend()
sns.move_legend(income_count, "upper left", bbox_to_anchor=(0.87, 0.935),\
                title=None, frameon=False)
leg = income_count._legend
leg.set_title('Impact')
plt.savefig('plots/impact_byincome_count.png', pad_inches= 0.15, dpi = 450)


income_perc = sns.FacetGrid(uod_filtered[uod_filtered['offer_type']!='informational'],\
                    col='gender',row='offer_type', col_order=['F', 'M', 'O'],\
                    margin_titles=True, sharex=False, sharey=True,\
                    gridspec_kws={"hspace":0.15})
income_perc = income_perc.map_dataframe(sns.histplot, stat ='percent',  x='income',\
                    common_norm = True, binrange=(20000,120000), binwidth=(5000),\
                    kde=True, hue='Impact interval', hue_order = impact_hue_order,\
                    palette='flare')
income_perc.set(xlabel ='Income ($)', ylabel = 'Percentage %')
for ax in income_perc.axes.flat:
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: '{:,.0f}'.format(x/1000) + 'K'))
income_perc.fig.suptitle('Impact by Income')
plt.savefig('plots/impact_byincome_perc.png', pad_inches= 0.15, dpi = 450)


#Plotting graphs for Impact across membership start
mem_count = sns.FacetGrid(uod_filtered, col='gender',row='offer_type',\
                    col_order=['F', 'M', 'O'], margin_titles=True,\
                    sharex=False, sharey=True, gridspec_kws={"hspace":0.15},\
                    legend_out=True, hue='Impact interval',\
                    hue_order = impact_hue_order, palette='flare')
mem_count = mem_count.map_dataframe(sns.histplot,  x='became_member_on',\
                    common_norm = True, bins=date2num(mem_bins),\
                    kde=True)
mem_count.set(xlabel ='Year', ylabel = 'Count')
mem_count.fig.subplots_adjust(top=0.9)
mem_count.fig.suptitle('Impact by Membership Start')
mem_count.add_legend()
sns.move_legend(mem_count, "upper left", bbox_to_anchor=(0.87, 0.935),\
                title=None, frameon=False)
leg = mem_count._legend
leg.set_title('Impact')
plt.savefig('plots/impact_bymembership_count.png', pad_inches= 0.15, dpi = 450)


mem_perc = sns.FacetGrid(uod_filtered[uod_filtered['offer_type']!='informational'],\
                    col='gender',row='offer_type', col_order=['F', 'M', 'O'],\
                    margin_titles=True, sharex=False, sharey=True,\
                    gridspec_kws={"hspace":0.15})
mem_perc = mem_perc.map_dataframe(sns.histplot, stat ='percent',  x='became_member_on',\
                    common_norm = True,  bins=date2num(mem_bins), kde=True,\
                    hue='Impact interval', hue_order = impact_hue_order,\
                    palette='flare')
mem_perc.set(xlabel ='Year', ylabel = 'Percentage %')
mem_perc.fig.suptitle('Impact by Membership Start')
plt.savefig('plots/impact_bymembership_perc.png', pad_inches= 0.15, dpi = 450)



#Looking at cumulative probability
age_ecdf = sns.displot(uod_filtered[(uod_filtered['offer_type']=='bogo') &\
                    ((uod_filtered['Impact interval']=='0.00 - 0.25') | (uod_filtered['Impact interval']=='0.75 - 1.00'))],\
                    x='age', kind ='ecdf',  hue='Impact interval',palette='flare')
age_ecdf.set(xlabel ='Age', ylabel = 'Probability', title = 'Cumulative Impact Probability by Age\n (across all genders)')

age_ecdf.fig.subplots_adjust(top=0.9) 
#age_ecdf.fig.suptitle('Age ECDF')
sns.move_legend(age_ecdf, "upper left", bbox_to_anchor=(0.8, 0.925), title=None, frameon=False)
#age_ecdf.add_legend()
leg = age_ecdf._legend
leg.set_title('Impact interval')
plt.savefig('plots/cumulative_impact_byage.png', pad_inches= 0.15, dpi = 450)


'''
Calculating a KS test score for comparison of two distributions under the
hypothesis that they are identical. The scipy.stat.kstest function will return 
a KS score and a p value. The test p-value can be set at p0=0.05 so if the KS
test returns p<p0 we can reject the null-hypothesis.
'''

#Looking at bogo offers
#Creating arrays of different subsets to use as samples for KS test
age_sample1 =uod_filtered[(uod_filtered['offer_type']=='bogo') & \
                           (uod_filtered['Impact interval']=='0.00 - 0.25')]['age'] 
age_sample2 =uod_filtered[(uod_filtered['offer_type']=='bogo') & \
                          (uod_filtered['Impact interval']=='0.75 - 1.00')]['age']

ageM_sample1 =uod_filtered[(uod_filtered['offer_type']=='bogo') & \
                           (uod_filtered['Impact interval']=='0.00 - 0.25') &\
                           (uod_filtered['gender']=='M')]['age'] 
ageM_sample2 =uod_filtered[(uod_filtered['offer_type']=='bogo') & \
                           (uod_filtered['Impact interval']=='0.75 - 1.00') &\
                           (uod_filtered['gender']=='M')]['age']

ageF_sample1 =uod_filtered[(uod_filtered['offer_type']=='bogo') & \
                           (uod_filtered['Impact interval']=='0.00 - 0.25') &\
                           (uod_filtered['gender']=='F')]['age'] 
ageF_sample2 =uod_filtered[(uod_filtered['offer_type']=='bogo') & \
                           (uod_filtered['Impact interval']=='0.75 - 1.00') &\
                           (uod_filtered['gender']=='F')]['age']

income_sample1 =uod_filtered[(uod_filtered['offer_type']=='bogo') & \
                             (uod_filtered['Impact interval']=='0.00 - 0.25')]['income'] 
income_sample2 =uod_filtered[(uod_filtered['offer_type']=='bogo') &\
                             (uod_filtered['Impact interval']=='0.75 - 1.00')]['income']

incomeM_sample1 =uod_filtered[(uod_filtered['offer_type']=='bogo') & \
                              (uod_filtered['Impact interval']=='0.00 - 0.25') &\
                              (uod_filtered['gender']=='M')]['income'] 
incomeM_sample2 =uod_filtered[(uod_filtered['offer_type']=='bogo') & \
                              (uod_filtered['Impact interval']=='0.75 - 1.00') &\
                              (uod_filtered['gender']=='M')]['income']

incomeF_sample1 =uod_filtered[(uod_filtered['offer_type']=='bogo') & \
                              (uod_filtered['Impact interval']=='0.00 - 0.25') &\
                           (uod_filtered['gender']=='F')]['income'] 
incomeF_sample2 =uod_filtered[(uod_filtered['offer_type']=='bogo') & \
                              (uod_filtered['Impact interval']=='0.75 - 1.00') &\
                           (uod_filtered['gender']=='F')]['income']

mem_sample1 =uod_filtered[(uod_filtered['offer_type']=='bogo') & \
                          (uod_filtered['Impact interval']=='0.00 - 0.25')]\
    ['became_member_on'] 
mem_sample2 =uod_filtered[(uod_filtered['offer_type']=='bogo') & \
                          (uod_filtered['Impact interval']=='0.75 - 1.00')]\
    ['became_member_on']    
    
memM_sample1 =uod_filtered[(uod_filtered['offer_type']=='bogo') & \
                           (uod_filtered['Impact interval']=='0.00 - 0.25') &\
                           (uod_filtered['gender']=='M')]['became_member_on'] 
memM_sample2 =uod_filtered[(uod_filtered['offer_type']=='bogo') & \
                           (uod_filtered['Impact interval']=='0.75 - 1.00') &\
                           (uod_filtered['gender']=='M')]['became_member_on']
    
memF_sample1 =uod_filtered[(uod_filtered['offer_type']=='bogo') & \
                           (uod_filtered['Impact interval']=='0.00 - 0.25') &\
                           (uod_filtered['gender']=='F')]['became_member_on'] 
memF_sample2 =uod_filtered[(uod_filtered['offer_type']=='bogo') & \
                           (uod_filtered['Impact interval']=='0.75 - 1.00') &\
                           (uod_filtered['gender']=='F')]['became_member_on']
    
    
#Creating KS tests    
KS_age = sp.stats.kstest(age_sample1, age_sample2)
KS_ageM = sp.stats.kstest(ageM_sample1, ageM_sample2)
KS_ageF = sp.stats.kstest(ageF_sample1, ageF_sample2)
KS_income = sp.stats.kstest(income_sample1, income_sample2)
KS_incomeM = sp.stats.kstest(incomeM_sample1, incomeM_sample2)
KS_incomeF = sp.stats.kstest(incomeM_sample1, incomeF_sample2)
KS_membership = sp.stats.kstest(mem_sample1, mem_sample2)
KS_membershipM = sp.stats.kstest(memM_sample1, memM_sample2)
KS_membershipF = sp.stats.kstest(memF_sample1, memF_sample2)

#Saving the results into DataFrame for easier result interpretation
KS_list = [KS_age, KS_ageM, KS_ageF, KS_income, KS_incomeM, KS_incomeF,\
           KS_membership, KS_membershipM, KS_membershipF]
KS_name = ['KS_age', 'KS_ageM','KS_ageF', 'KS_income', 'KS_incomeM',\
           'KS_incomeF', 'KS_membership', 'KS_membershipM', 'KS_membershipF']

KS_test = pd.DataFrame(KS_list, KS_name).reset_index()
KS_test.rename(columns={'index': 'test', }, inplace=True)
KS_test.sort_values('pvalue', inplace=True)
KS_test['offer type'] = 'bogo'


#Looking at discount offers
#Creating arrays of different subsets to use as samples for KS test
age2_sample1 =uod_filtered[(uod_filtered['offer_type']=='discount') & \
                           (uod_filtered['Impact interval']=='0.00 - 0.25')]['age'] 
age2_sample2 =uod_filtered[(uod_filtered['offer_type']=='discount') & \
                          (uod_filtered['Impact interval']=='0.75 - 1.00')]['age']

ageM2_sample1 =uod_filtered[(uod_filtered['offer_type']=='discount') & \
                           (uod_filtered['Impact interval']=='0.00 - 0.25') &\
                           (uod_filtered['gender']=='M')]['age'] 
ageM2_sample2 =uod_filtered[(uod_filtered['offer_type']=='discount') & \
                           (uod_filtered['Impact interval']=='0.75 - 1.00') &\
                           (uod_filtered['gender']=='M')]['age']

ageF2_sample1 =uod_filtered[(uod_filtered['offer_type']=='discount') & \
                           (uod_filtered['Impact interval']=='0.00 - 0.25') &\
                           (uod_filtered['gender']=='F')]['age'] 
ageF2_sample2 =uod_filtered[(uod_filtered['offer_type']=='discount') & \
                           (uod_filtered['Impact interval']=='0.75 - 1.00') &\
                           (uod_filtered['gender']=='F')]['age']

income2_sample1 =uod_filtered[(uod_filtered['offer_type']=='discount') & \
                             (uod_filtered['Impact interval']=='0.00 - 0.25')]['income'] 
income2_sample2 =uod_filtered[(uod_filtered['offer_type']=='discount') &\
                             (uod_filtered['Impact interval']=='0.75 - 1.00')]['income']

incomeM2_sample1 =uod_filtered[(uod_filtered['offer_type']=='discount') & \
                              (uod_filtered['Impact interval']=='0.00 - 0.25') &\
                              (uod_filtered['gender']=='M')]['income'] 
incomeM2_sample2 =uod_filtered[(uod_filtered['offer_type']=='discount') & \
                              (uod_filtered['Impact interval']=='0.75 - 1.00') &\
                              (uod_filtered['gender']=='M')]['income']

incomeF2_sample1 =uod_filtered[(uod_filtered['offer_type']=='discount') & \
                              (uod_filtered['Impact interval']=='0.00 - 0.25') &\
                           (uod_filtered['gender']=='F')]['income'] 
incomeF2_sample2 =uod_filtered[(uod_filtered['offer_type']=='discount') & \
                              (uod_filtered['Impact interval']=='0.75 - 1.00') &\
                           (uod_filtered['gender']=='F')]['income']

mem2_sample1 =uod_filtered[(uod_filtered['offer_type']=='discount') & \
                          (uod_filtered['Impact interval']=='0.00 - 0.25')]\
    ['became_member_on'] 
mem2_sample2 =uod_filtered[(uod_filtered['offer_type']=='discount') & \
                          (uod_filtered['Impact interval']=='0.75 - 1.00')]\
    ['became_member_on']    
    
memM2_sample1 =uod_filtered[(uod_filtered['offer_type']=='discount') & \
                           (uod_filtered['Impact interval']=='0.00 - 0.25') &\
                           (uod_filtered['gender']=='M')]['became_member_on'] 
memM2_sample2 =uod_filtered[(uod_filtered['offer_type']=='discount') & \
                           (uod_filtered['Impact interval']=='0.75 - 1.00') &\
                           (uod_filtered['gender']=='M')]['became_member_on']
    
memF2_sample1 =uod_filtered[(uod_filtered['offer_type']=='discount') & \
                           (uod_filtered['Impact interval']=='0.00 - 0.25') &\
                           (uod_filtered['gender']=='F')]['became_member_on'] 
memF2_sample2 =uod_filtered[(uod_filtered['offer_type']=='discount') & \
                           (uod_filtered['Impact interval']=='0.75 - 1.00') &\
                           (uod_filtered['gender']=='F')]['became_member_on']
    
#age_sample1
#age2_sample1

    
#Creating KS tests    
KS_age2 = sp.stats.kstest(age2_sample1, age2_sample2)
KS_ageM2 = sp.stats.kstest(ageM2_sample1, ageM2_sample2)
KS_ageF2 = sp.stats.kstest(ageF2_sample1, ageF2_sample2)
KS_income2 = sp.stats.kstest(income2_sample1, income2_sample2)
KS_incomeM2 = sp.stats.kstest(incomeM2_sample1, incomeM2_sample2)
KS_incomeF2 = sp.stats.kstest(incomeM2_sample1, incomeF2_sample2)
KS_membership2 = sp.stats.kstest(mem2_sample1, mem2_sample2)
KS_membershipM2 = sp.stats.kstest(memM2_sample1, memM2_sample2)
KS_membershipF2 = sp.stats.kstest(memF2_sample1, memF2_sample2)

KS_list2 = [KS_age2, KS_ageM2, KS_ageF2, KS_income2, KS_incomeM2, KS_incomeF2,\
            KS_membership2, KS_membershipM2, KS_membershipF2]
    
KS_test2 = pd.DataFrame(KS_list2, KS_name).reset_index()
KS_test2.rename(columns={'index': 'test', }, inplace=True)
KS_test2.sort_values('pvalue', inplace=True)
KS_test2['offer type'] = 'discount'



#Combining the KS test results for both offer types
KS_test = pd.concat([KS_test, KS_test2], ignore_index=True)
KS_test.sort_values(by = ['pvalue','test'], inplace=True)
KS_test.style.bar(subset=['statistics'], align='mid', color=['#fec20c', '#00ab41'])\
    .format({'statistics': '{:,.2%}'.format,'pvalue': '{:,.2%}'.format})
display(KS_test)    
    
#Saving KS test results to a separate csv file
KS_test.to_csv('KS_test.csv')


#Comparing successfull offers amongs two different offer types
#Creating arrays of different subsets to use as samples for KS test
age3_sample1 =uod_filtered[(uod_filtered['offer_type']=='bogo') & \
                           (uod_filtered['Impact interval']=='0.75 - 1.00')]['age'] 
age3_sample2 =uod_filtered[(uod_filtered['offer_type']=='discount') & \
                          (uod_filtered['Impact interval']=='0.75 - 1.00')]['age']

ageM3_sample1 =uod_filtered[(uod_filtered['offer_type']=='bogo') & \
                           (uod_filtered['Impact interval']=='0.75 - 1.00') &\
                           (uod_filtered['gender']=='M')]['age'] 
ageM3_sample2 =uod_filtered[(uod_filtered['offer_type']=='discount') & \
                           (uod_filtered['Impact interval']=='0.75 - 1.00') &\
                           (uod_filtered['gender']=='M')]['age']

ageF3_sample1 =uod_filtered[(uod_filtered['offer_type']=='bogo') & \
                           (uod_filtered['Impact interval']=='0.75 - 1.00') &\
                           (uod_filtered['gender']=='F')]['age'] 
ageF3_sample2 =uod_filtered[(uod_filtered['offer_type']=='discount') & \
                           (uod_filtered['Impact interval']=='0.75 - 1.00') &\
                           (uod_filtered['gender']=='F')]['age']

income3_sample1 =uod_filtered[(uod_filtered['offer_type']=='bogo') & \
                             (uod_filtered['Impact interval']=='0.75 - 1.00')]['income'] 
income3_sample2 =uod_filtered[(uod_filtered['offer_type']=='discount') &\
                             (uod_filtered['Impact interval']=='0.75 - 1.00')]['income']

incomeM3_sample1 =uod_filtered[(uod_filtered['offer_type']=='bogo') & \
                              (uod_filtered['Impact interval']=='0.75 - 1.00') &\
                              (uod_filtered['gender']=='M')]['income'] 
incomeM3_sample2 =uod_filtered[(uod_filtered['offer_type']=='discount') & \
                              (uod_filtered['Impact interval']=='0.75 - 1.00') &\
                              (uod_filtered['gender']=='M')]['income']

incomeF3_sample1 =uod_filtered[(uod_filtered['offer_type']=='bogo') & \
                              (uod_filtered['Impact interval']=='0.75 - 1.00') &\
                           (uod_filtered['gender']=='F')]['income'] 
incomeF3_sample2 =uod_filtered[(uod_filtered['offer_type']=='discount') & \
                              (uod_filtered['Impact interval']=='0.75 - 1.00') &\
                           (uod_filtered['gender']=='F')]['income']

mem3_sample1 =uod_filtered[(uod_filtered['offer_type']=='bogo') & \
                          (uod_filtered['Impact interval']=='0.75 - 1.00')]\
    ['became_member_on'] 
mem3_sample2 =uod_filtered[(uod_filtered['offer_type']=='discount') & \
                          (uod_filtered['Impact interval']=='0.75 - 1.00')]\
    ['became_member_on']    
    
memM3_sample1 =uod_filtered[(uod_filtered['offer_type']=='bogo') & \
                           (uod_filtered['Impact interval']=='0.75 - 1.00') &\
                           (uod_filtered['gender']=='M')]['became_member_on'] 
memM3_sample2 =uod_filtered[(uod_filtered['offer_type']=='discount') & \
                           (uod_filtered['Impact interval']=='0.75 - 1.00') &\
                           (uod_filtered['gender']=='M')]['became_member_on']
    
memF3_sample1 =uod_filtered[(uod_filtered['offer_type']=='bogo') & \
                           (uod_filtered['Impact interval']=='0.75 - 1.00') &\
                           (uod_filtered['gender']=='F')]['became_member_on'] 
memF3_sample2 =uod_filtered[(uod_filtered['offer_type']=='discount') & \
                           (uod_filtered['Impact interval']=='0.75 - 1.00') &\
                           (uod_filtered['gender']=='F')]['became_member_on']
    
#age_sample1
#age2_sample1

    
#Creating KS tests    
KS_age3 = sp.stats.kstest(age3_sample1, age3_sample2)
KS_ageM3 = sp.stats.kstest(ageM3_sample1, ageM3_sample2)
KS_ageF3 = sp.stats.kstest(ageF3_sample1, ageF3_sample2)
KS_income3 = sp.stats.kstest(income3_sample1, income3_sample2)
KS_incomeM3 = sp.stats.kstest(incomeM3_sample1, incomeM3_sample2)
KS_incomeF3 = sp.stats.kstest(incomeM3_sample1, incomeF3_sample2)
KS_membership3 = sp.stats.kstest(mem3_sample1, mem3_sample2)
KS_membershipM3 = sp.stats.kstest(memM3_sample1, memM3_sample2)
KS_membershipF3 = sp.stats.kstest(memF3_sample1, memF3_sample2)

KS_list3 = [KS_age3, KS_ageM3, KS_ageF3, KS_income3, KS_incomeM3, KS_incomeF3,\
            KS_membership3, KS_membershipM3, KS_membershipF3]
    
KS_test_offercomp = pd.DataFrame(KS_list3, KS_name).reset_index()
KS_test_offercomp.rename(columns={'index': 'test', }, inplace=True)
KS_test_offercomp.sort_values('pvalue', inplace=True)

display(KS_test_offercomp)
KS_test_offercomp.to_csv('KS_test_offercomp.csv')

print('\n All plots as well as the KS test results have been saved in separate files. Plots can be found in \'plots/\' folder as .png and KS results as .csv files.') 