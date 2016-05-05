import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

trains = pd.read_csv('./train_users_2.csv')
print(trains.info())
print(trains.head())
plt.style.use(['seaborn-whitegrid', 'seaborn-pastel'])

trains.age.hist(bins=50)
trains[trains.age > 1900].age.hist(bins=5)
import seaborn as sns
trains.loc[trains.age > 100, 'age'] = np.nan
trains.loc[trains.age < 0, 'age'] = np.nan
trains.age.dropna().hist(bins=20)

plt.style.use(['seaborn-whitegrid', 'seaborn-deep'])

trains.gender.value_counts().plot(kind='bar',rot=0)
trains.language.value_counts().plot(kind='bar')

trains.signup_app.value_counts().plot(kind='bar')
trains.signup_method.value_counts().plot(kind='bar')

trains.affiliate_channel.value_counts().plot(kind='bar')
trains.first_affiliate_tracked.value_counts().plot(kind='bar',rot=0)

trains.first_device_type.value_counts().plot(kind='bar',rot=0)
trains.first_browser.value_counts().plot(kind='bar')

trains['date_account_created'] = pd.to_datetime(trains['date_account_created'])
trains.date_account_created.value_counts().plot(kind='line',linewidth=1)

trains['date_first_active'] = pd.to_datetime((trains.timestamp_first_active//1000000),format='%Y%m%d')
trains.date_first_active.value_counts().plot(kind='line',linewidth=1)

trains['date_first_booking'] = pd.to_datetime(trains['date_first_booking'])
trains.date_first_booking.value_counts().plot(kind='line',color='#FD5c66',linewidth=1)

dac_splited = np.vstack(trains.date_account_created.apply(lambda x: list(map(int,x.split('-')))).values)
trains['dac_year'] = dac_splited[:,0]
trains['dac_month'] = dac_splited[:,1]
trains['dac_day'] = dac_splited[:,2]

trains.country_destination.value_counts().plot(kind='bar',color='#FD5c66',rot=0)
