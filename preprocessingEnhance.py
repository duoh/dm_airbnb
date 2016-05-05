import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date

data = pd.read_csv('/Users/duoh/Documents/Mahidol/Data Mining/project/dataset/train_users_2.csv')

data = data.drop(['id','date_first_booking'], axis=1)

age_data = data.age.values
data['age'] = np.where(np.logical_or(age_data<=0, age_data>100), -1, age_data)

#AgeRange
#(One-hot encoding of the edge according these intervals)
interv =  [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
def get_interv_value(age):
    iv = 0
    for i in range(len(interv)):
        if age < interv[i]:
            iv = i
            break
    return iv


data['age_interv'] = data.age.apply(lambda x: get_interv_value(x))
df_tt_ai = pd.get_dummies(data.age_interv, prefix='age_interv')
data = data.drop(['age_interv'], axis=1)
data = pd.concat((data, df_tt_ai), axis=1)
data = data.drop(['age'], axis=1)

data['date_first_active'] = pd.to_datetime((data.timestamp_first_active//1000000),format='%Y%m%d')
data = data.drop(['timestamp_first_active'],axis=1)
dfa_splited = np.vstack(data.date_first_active.astype(str).apply(lambda x: list(map(int,x.split('-')))).values)
data['dfa_year'] = dfa_splited[:,0]
data['dfa_month'] = dfa_splited[:,1]
data['dfa_day'] = dfa_splited[:,2]
data['dfa_weekday'] = np.vstack(data.date_first_active.apply(lambda x: x.weekday()))
data['dfa_week'] = np.vstack(data.date_first_active.apply(lambda x: x.isocalendar()[1]))

dac_splited = np.vstack(data.date_account_created.apply(lambda x: list(map(int,x.split('-')))).values)
data['dac_year'] = dac_splited[:,0]
data['dac_month'] = dac_splited[:,1]
data['dac_day'] = dac_splited[:,2]
data['date_account_created'] = pd.to_datetime(data['date_account_created'])
data['dac_weekday'] = np.vstack(data.date_account_created.apply(lambda x: x.weekday()))
data['dac_week'] = np.vstack(data.date_account_created.apply(lambda x: x.isocalendar()[1]))

Y = 2000
seasons = [(0, (date(Y,  1,  1),  date(Y,  3, 20))),  #'winter'
           (1, (date(Y,  3, 21),  date(Y,  6, 20))),  #'spring'
           (2, (date(Y,  6, 21),  date(Y,  9, 22))),  #'summer'
           (3, (date(Y,  9, 23),  date(Y, 12, 20))),  #'autumn'
           (0, (date(Y, 12, 21),  date(Y, 12, 31)))]  #'winter'
def get_season(dt):
    dt = dt.date()
    dt = dt.replace(year=Y)
    return next(season for season, (start, end) in seasons if start <= dt <= end)


data['dac_season'] = np.array([get_season(dt) for dt in data.date_account_created])
data['dfa_season'] = np.array([get_season(dt) for dt in data.date_first_active])

data = data.drop(['date_account_created'],axis=1)
data = data.drop(['date_first_active'],axis=1)

data = data.fillna(-1)

data['num_missing'] = np.array([sum(r == -1) for r in data.values])

apple = []
for i in range(len(data)):
    if ( data['signup_app'][i] == 'iOS' or data['signup_app'][i] == 'Moweb' ) and \
        ( data['first_device_type'][i] == 'iPad' or data['first_device_type'][i] == 'iPhone' or \
         data['first_device_type'][i] == 'Mac Desktop') and \
        ( data['first_browser'][i] == 'Safari' or data['first_browser'][i] == 'Mobile Safari'):
        apple.append(1)
    else:
        apple.append(0)

data['apple_loyal'] = apple

google = []
for i in range(len(data)):
    if ( data['signup_app'][i] == 'Android' or data['signup_app'][i] == 'Moweb' ) and \
        ( data['first_device_type'][i] == 'Android Tablet' or data['first_device_type'][i] == 'Android Phone') and \
        ( data['first_browser'][i] == 'Chrome' or data['first_browser'][i] == 'Android Browser'):
        google.append(1)
    else:
        google.append(0)

data['google_loyal'] = google

onehot_features = ['gender','signup_method','signup_flow','language','affiliate_channel','affiliate_provider','first_affiliate_tracked','signup_app','first_device_type','first_browser']
for feature in onehot_features:
    dummy = pd.get_dummies(data[feature], prefix=feature)
    data = data.drop([feature], axis=1)
    data = pd.concat((data, dummy), axis=1)

data.to_csv('/Users/duoh/Documents/Mahidol/Data Mining/project/preprocessDataEnhance.csv',index=False)