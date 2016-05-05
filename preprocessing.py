import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/duoh/Documents/Mahidol/Data Mining/project/dataset/train_users_2.csv')

data = data.drop(['id','date_first_booking'], axis=1)

age_data = data.age.values
data['age'] = np.where(np.logical_or(age_data<18, age_data>100), -1, age_data)

data['date_first_active'] = pd.to_datetime((data.timestamp_first_active//1000000),format='%Y%m%d')
data = data.drop(['timestamp_first_active'],axis=1)
dfa_splited = np.vstack(data.date_first_active.astype(str).apply(lambda x: list(map(int,x.split('-')))).values)
data['dfa_year'] = dfa_splited[:,0]
data['dfa_month'] = dfa_splited[:,1]
data['dfa_day'] = dfa_splited[:,2]
data['dfa_weekday'] = np.vstack(data.date_first_active.apply(lambda x: x.weekday()))
data = data.drop(['date_first_active'],axis=1)

dac_splited = np.vstack(data.date_account_created.apply(lambda x: list(map(int,x.split('-')))).values)
data['dac_year'] = dac_splited[:,0]
data['dac_month'] = dac_splited[:,1]
data['dac_day'] = dac_splited[:,2]
data['date_account_created'] = pd.to_datetime(data['date_account_created'])
data['dac_weekday'] = np.vstack(data.date_account_created.apply(lambda x: x.weekday()))
data = data.drop(['date_account_created'],axis=1)

data = data.fillna(-1)

onehot_features = ['gender','signup_method','signup_flow','language','affiliate_channel','affiliate_provider','first_affiliate_tracked','signup_app','first_device_type','first_browser']

for feature in onehot_features:
    dummy = pd.get_dummies(data[feature], prefix=feature)
    data = data.drop([feature], axis=1)
    data = pd.concat((data, dummy), axis=1)

data.to_csv('/Users/duoh/Documents/Mahidol/Data Mining/project/preprocessData.csv',index=False)