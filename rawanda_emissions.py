# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:36:15 2023

@author: codyt
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras import backend as K
from sklearn.cluster import KMeans

train_file = '.\\train.csv'
test_file = '.\\test.csv'

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

train = train.sort_values(by=['latitude','longitude','year','week_no'])

head = train.head()

# latlong_count = train.groupby(['latitude','longitude'], as_index=False).count()['week_no']
latlong_count = train.groupby(['latitude','longitude'], as_index=False).agg({'week_no':'count', 'emission':['max','mean','std']})
latlong_count = latlong_count.sort_values(('emission','mean'), ascending=False)

kmeans_emission = KMeans(n_clusters=7)
latlong_count['kmeans_emission_group'] = kmeans_emission.fit_predict(latlong_count[('emission','mean')].to_numpy().reshape(-1,1))
kmeans_location = KMeans(n_clusters=10)

kmeans_location.fit(latlong_count[[('latitude',''),('longitude','')]])
labels = kmeans_location.predict(latlong_count[[('latitude',''),('longitude','')]])
centroids  = kmeans_location.cluster_centers_
[centroids[i] for i in labels]
latlong_count['kmeans_location_group'] = labels




def datetimeindex_from_year_weekno(df):
    ''' needs a df with columns 'year' and 'week_no' '''
    df['date_str'] = df['year'].astype(str) + '-W' + df['week_no'].astype(str)

    df_index = pd.to_datetime(df['date_str'].astype(str) + '-1', format='%Y-W%U-%w') 
    df['index'] = df.index
    return df.set_index(df_index, drop=False)


train = datetimeindex_from_year_weekno(train)
test = datetimeindex_from_year_weekno(test)

na_count1 = train.isna().sum()

def interpolate_by_group(df):
    coords = df[['latitude','longitude']].drop_duplicates()
    interpolated_df = []
    for _, coord_row in coords.iterrows():
        lat = coord_row['latitude']
        long = coord_row['longitude']
        subset = df[(df['latitude'] == lat) & (df['longitude'] == long)]
        # this should change it in df as well
        subset = subset.interpolate(method='time')
        interpolated_df.append(subset)
    return pd.concat(interpolated_df)   

train = interpolate_by_group(train) 
           
na_count2 = train.isna().sum()      
train = train.interpolate(method='linear')
na_count3 = train.isna().sum()
  
train = train.set_index(['index'])
test = test.set_index(['index'])

def put_kmeans_groups_onto_df(df):
    df = pd.merge(df, latlong_count[['latitude','longitude','kmeans_emission_group','kmeans_location_group']], left_on=['latitude','longitude'], right_on=[('latitude',''),('longitude','')], how='left')
    df = df.drop(axis='columns', columns=[('latitude', ''),('longitude', '')])
    df = df.rename(mapper={('kmeans_emission_group', ''):'kmeans_emission_group',
                                 ('kmeans_location_group', ''):'kmeans_location_group'},
                   axis='columns')

    return df
    
train = put_kmeans_groups_onto_df(train)
train_head = train.head()
test = put_kmeans_groups_onto_df(test)

# interpolate with time method
# train = train.interpolate(method='time')
# interpolate with linear method
# train.interpolate(method='linear')

plt.matshow(train.corr())

correlation_matrix = train.corr(method='pearson')['emission']
correlation_matrix = correlation_matrix.drop(['latitude','longitude','emission','week_no'])
# correlation_matrix = correlation_matrix.drop(['emission','week_no'])
other_columns_to_drop = [i for i in list(correlation_matrix.index) if 'angle' in i or 'azimuth' in i or 'zenith' in i]
correlation_matrix = correlation_matrix.drop(other_columns_to_drop)
correlation_matrix = correlation_matrix.sort_values(ascending=False)



columns_necisito = ['ID_LAT_LON_YEAR_WEEK','year','week_no', 'latitude','longitude']
columns_regressors = list(correlation_matrix.index[:20])
target = 'emission'

train1 = train[columns_necisito + columns_regressors + [target]]
train1 = train1.sort_values(['year','week_no','latitude','longitude'])
train1 = train1.dropna()

scaler_regressor = StandardScaler()
scaler_target = StandardScaler()
# init the df
train_normalized = train1.copy()
# normalize the regressor columns
train_normalized[columns_regressors] = scaler_regressor.fit_transform(train1[columns_regressors])
train_normalized[target] = scaler_target.fit_transform(train1[target].to_numpy().reshape(-1,1))

X = train_normalized[columns_necisito + columns_regressors]
y = train_normalized[columns_necisito + [target]]


train_denormalized = train_normalized.copy()
train_denormalized[columns_regressors] = scaler_regressor.inverse_transform(train_normalized[columns_regressors])
train_denormalized[target] = scaler_target.inverse_transform(train_normalized[target].to_numpy().reshape(-1,1))



split_fraction = 0.2


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_fraction, random_state = 69, shuffle=False)

print(X_train.shape)
print(X_test.shape)


#%%
sequence_length = 1
learning_rate = 0.001
epochs = 1000
batch_size = 256


dataset_train = keras.preprocessing.timeseries_dataset_from_array(X_train[columns_regressors], y_train[[target]], sequence_length, batch_size=batch_size)
dataset_val = keras.preprocessing.timeseries_dataset_from_array(X_test[columns_regressors], y_test[[target]], sequence_length, batch_size=batch_size)

# features = X_train[['latitude', 'longitude']].to_numpy()
# target = y_train[target].to_numpy()
# dataset_train = keras.preprocessing.timeseries_dataset_from_array(X_train[columns_regressors].to_numpy(),  
#                                                                   target, 
#                                                                   sequence_length=sequence_length,  
#                                                                   batch_size=batch_size)



for batch in dataset_train.take(1):
    inputs, targets = batch
    
print(inputs.numpy().shape)
print(targets.numpy().shape)


#%%

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))



inputs = keras.layers.Input(shape = (inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.LSTM(32)(inputs)
outputs = keras.layers.Dense(1)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)
# model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['accuracy'])
model.compile(optimizer='rmsprop', loss=root_mean_squared_error, metrics =[])
model.summary()




#%%

path_checkpoint = 'model_checkpoint.h5'
es_callback = keras.callbacks.EarlyStopping(patience=5)

modelckpt_callback = keras.callbacks.ModelCheckpoint(filepath = path_checkpoint, 
                                                     monitor ='val_loss',
                                                     verbose = 1,
                                                     save_weights_only = True,
                                                     save_best_only = True)

history = model.fit(x = dataset_train, 
                    epochs = epochs, 
                    validation_data = dataset_val, 
                    callbacks = [es_callback, modelckpt_callback])


#%%

train2 = train1.copy()
train2['train'] = 1
test1 = test[columns_necisito + columns_regressors]
test1['train'] = 0
test1[target] = np.nan
test_and_train = pd.concat([train2, test1])

na_count1 = test_and_train.isna().sum()
test_and_train = datetimeindex_from_year_weekno(test_and_train)
test_and_train = interpolate_by_group(test_and_train) 
na_count2 = test_and_train.isna().sum()
test_and_train = test_and_train.interpolate(method='linear')
na_count3 = test_and_train.isna().sum()


test2 = test_and_train[test_and_train['train'] == 0]
test2 = test2.drop(['train'], axis='columns')
test2 = test2.set_index('index', drop=True)
test2_normalized = test2.copy()
test2_normalized[columns_regressors] = scaler_regressor.fit_transform(test2[columns_regressors])
# test2_normalized[target] = scaler_target.fit_transform(test2[target].to_numpy().reshape(-1,1))
# test2[columns_regressors].to_numpy()
# test2[target]

dataset_test = keras.preprocessing.timeseries_dataset_from_array(test2_normalized[columns_regressors].to_numpy(), targets=None,
                                                                 sequence_length=sequence_length, batch_size=batch_size)

pred_normalized = model.predict(dataset_test)

pred_denormalized = scaler_target.inverse_transform(pred_normalized)

output = test2[columns_necisito]
output[target] = pred_denormalized

output = output.sort_index()

output[['ID_LAT_LON_YEAR_WEEK','emission']].to_csv('.\\output3.csv', index=False)
