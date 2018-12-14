#%%
import data.data as data
import inout.importexport as imp
import numpy as np
import pandas as pd
import xgboost as xgb
import random
import math
from pandas.api.types import CategoricalDtype

#%% [markdown]
# ### Create the complete dataframe
#%% [markdown]
# #### Read the recommendations from csv

#%%
raw_recs = imp.importcsv('submission/13-12-2018/gxboost25recommendations_12-17-23.csv', check_len=-1)


#%% [markdown]
# #### Explode each row into multiple rows (one per interaction)

#%%
recs_tracks = []
for rec in raw_recs:
    playlist_id = rec[0]
    for t in rec[1:]:
        recs_tracks.append([playlist_id, t])
recs_df = pd.DataFrame(recs_tracks, columns=['playlist_id','track_id'])

#%% [markdown]
# #### Append the 'profile_length' column to the recommendation dataframe

#%%
target_ids = data.get_target_playlists()
targetURM = data.get_urm_train_1()[target_ids]
user_profile_lengths = np.array(targetURM.sum(axis=1)).flatten()
profile_lengths_df = pd.DataFrame({'playlist_id': target_ids, 'profile_length': user_profile_lengths})

#%%
rec_lengths_df = recs_df.merge(profile_lengths_df, on='playlist_id')

#%% [markdown]
# #### Popularity feature

#%%
df = data.get_playlists_df()
popularity = df.groupby(['track_id']).size().reset_index(name='popularity')

#%%
rec_pop_df = rec_lengths_df.join(popularity.set_index('track_id'), on='track_id')

#%% [markdown]
# #### Append the 'label' column 

#%%
urm_test = data.get_urm_test_1()
test_labels = []

last_playlist_id = -1
for idx,row in recs_df.iterrows():
    current_playlist_id = row['playlist_id']
    track_id = row['track_id']
    # cache the row of the urm test if same playlist of the previous iteration
    if not current_playlist_id == last_playlist_id:
        # tracks ids in the t row of urm test
        tracks_ids = urm_test.getrow(current_playlist_id).nonzero()[1]
        last_playlist_id = current_playlist_id
    
    test_labels.append(1 if track_id in tracks_ids else 0)

test_labels_df = pd.DataFrame({'label': test_labels})


#%%
rec_label_df = pd.concat([rec_pop_df, test_labels_df], axis=1)

#%% [markdown]
# #### Append the tracks features (album, artist, duration)

#%%
tdf = data.get_tracks_df()
rec_feature_track_df = rec_label_df.join(tdf.set_index('track_id'), on='track_id')

#%% [markdown]
# ### I'm happy with the features gathered

#%%
feature_df = rec_feature_track_df

#%% [markdown]
# ### Split into train and test dataframes

#%%
def func(x):
    n = x['label'].sum()
    ones = x.loc[x['label'] == 1]
    zeros = x.loc[x['label'] == 0].sample(n)
    return pd.concat([ones,zeros])

#%%
full = feature_df.groupby(['playlist_id'], as_index=False).apply(func)


#%%
full = full.reset_index().drop(['level_0', 'level_1'], axis=1)


#%%
tgt = data.get_target_playlists()
train_tgt = random.sample(tgt, math.floor(len(tgt)*0.8))
test_tgt = list(set(tgt) - set(train_tgt))
train = full.loc[full['playlist_id'].isin(train_tgt)]
test = full.loc[full['playlist_id'].isin(test_tgt)]

#%% [markdown]
# #### One hot encodings

#%%
to_concat_train = []
to_concat_test = []
to_onehot = ['album_id', 'artist_id', 'track_id', 'playlist_id']


#%%
def onehotize(full, df, str):
    exp = full[str].unique()
    print(len(exp))
    df.loc[:, (str)] = df[str].astype(CategoricalDtype(categories = exp))
    oh = pd.get_dummies(df[str], prefix=str).to_sparse(fill_value=0)
    return oh


#%%
for name in to_onehot:
    to_concat_train.append(onehotize(full, train, name))
    to_concat_test.append(onehotize(full, test, name))
    train = train.drop(name, axis=1)
    test = test.drop(name, axis=1)
to_concat_train.insert(0, train)
to_concat_test.insert(0, test)


#%%
train = pd.concat(to_concat_train, axis=1)
test = pd.concat(to_concat_test, axis=1)

print('data are ready!')

#%%
label_train = train.label
train = train.drop(['label'], axis=1)
label_test = test.label
test = test.drop(['label'], axis=1)
dtrain = xgb.DMatrix(train, label=label_train, missing=0)
test = xgb.DMatrix(test, label=label_test, missing=0)

param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'bpr'

evallist = []

num_round = 10
model_trained = xgb.train(param, dtrain, num_round, evallist)