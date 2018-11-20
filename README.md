# Recommender System 2018 Challenge Polimi
<p align="center">
  <img width="100%" src="https://i.imgur.com/tm9mSuM.png" alt="Recommender System 2018 Challenge Polimi">
</p>

## 1. Preprocessing

If you want to use the original csv of the competition, run the following script to preprocess the data and save the matrices used in the various algorithms for later usage:

    python preprocessing/urm.py

Matrices are saved into *raw_data/matrices*.
If you have your own, dataset, you can skip this step.


## 2. Data

The module `Data` is the main wrapper between to access the dataset saved in the disk. It is used to load the csv files and the .npz files as arrays or sparse matrices or dataframes (depending on the type of usage).

### Constants:
```python
N_PLAYLISTS = 50446
N_TRACKS = 20635
N_ARTISTS = 6668
N_ALBUMS = 12744
```

### Matrices:
```python
get_urm()
get_urm_train()
get_urm_test()
get_icm()
```
### Dataframes (Pandas):
```python
get_tracks_df()
get_playlists_df()
```

### Arrays:
```python
get_target_playlists()
get_all_playlists()
```

If you want to adapt your own dataset, you have to rename your files so that they reflect the paths of this class.


## 3. Recommenders

All the implemented recommenders are inside the following folder: `recommenders`

They inherit from a common base class called `RecommenderBase`. This abstract class exposes some useful methods, like:
```python
fit()
```
Fit the model on the data. Inherited class should extend this method in the appropriate way.

```python
recommend(self, userid, N=10, urm=None, filter_already_liked=True, with_scores=False, items_to_exclude=[])
```
Compute the N best items for the specified user.

```python
recommend_batch(self, userids, N=10, urm=None, filter_already_liked=True, with_scores=False, items_to_exclude=[], verbose=False)
```
Recommend the N best items for the specified list of users.

```python
evaluate(self, recommendations, test_urm, at_k=10)
```
Return the MAP@k (default MAP10) evaluation for the provided recommendations
computed with respect to the test_urm.

## Run the models

To run the models, check the various files inside: `run`.

Each file contains a `run` and a `test` function: `test` only calls `run` in verbose mode, without saving any result into the disk, useful to try a model and see how it behaves
