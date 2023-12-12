###import Packages
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

###Accessing NFL Data
project_dir = "C:/Users/Justin/Documents/projects/NFL project"
season_data = os.listdir(f"{project_dir}/nfl data")

#print(season_data)

data_files = ([f"""{project_dir}/nfl data/{csv_file}""" 
               for csv_file in season_data])
#print(data_files)

###Initialize data frame
df = pd.DataFrame()

###append to dataframe
for i in data_files:
    df = pd.concat([df,pd.read_csv(i)])
###print(df.shape)

###print(df.sample(3))
#print(df.columns.values)

###Picking out relevent QB data
qb_feats = ['season', 'passer_id','passer','pass','complete_pass','interception','sack','yards_gained','touchdown']
groupby_feats = ['season','passer_id','passer']

###combining relevant data in another dataframe
qb_df = (df.loc[:,qb_feats].groupby(groupby_feats, as_index=False).sum())

#print(qb_df.sample(10))
for y in ['yards_gained','complete_pass','pass','interception','sack']:
    sns.regplot(data=qb_df, x='touchdown', y=y)
    plt.title(f"Touchdown in {y}")
##copying dataframe to compare seasons 
_df = qb_df.copy()
##adding 1 to seasons
_df['season'] = _df['season'].add(1)
print(type(_df['season']))

new_qb_df = (qb_df.merge(_df, on=['season','passer_id','passer'],suffixes=('','_prev'), how="left"))
print(new_qb_df.sample(10))

###finding correlation with next season
for y in ['touchdown_prev','yards_gained_prev','complete_pass_prev','pass_prev','interception_prev','sack_prev']:
    sns.regplot(data=new_qb_df, x='touchdown', y=y)
    plt.title(f"touchdowns in {y}")
#Selecting key components that correlate to touchdowns
features = ['pass_prev','complete_pass_prev','interception_prev','sack_prev','yards_gained_prev','touchdown_prev']
target = 'touchdown'
model_data = new_qb_df.dropna(subset= features +[target])
model = LinearRegression()