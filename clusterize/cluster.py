"""
Functions for partition the train dataset based on different criteria
and visualize some (maybe) useful characteristics.
"""

import data.data as data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def cluster_users_by_interactions_count(clip):
    """
    Split the playlists based on interactions count above or below the specified clip value.
    
    Parameters
    ----------
    clip : (int), clip value for splitting. The playlists will be splitted in 2 groups:
        those that have an interactions count <= clip and those that have an interactions count > clip.

    Returns
    -------
    2 lists of playlists ids
    """
    playlists = data.get_playlists_df()
    
    # build dataframe of number of interactions: playlist_id | tracks_count
    counts = playlists.groupby('playlist_id').size().reset_index(name='counts')

    # split based on the interactions counts
    return counts[counts['counts']<=clip]['playlist_id'].values, counts[counts['counts']>clip]['playlist_id'].values


def histogram_of_interactions():
    """
    Plot the histogram of the interactions counts:
    x axis: interactions
    y axis: count of playlists with that number of interactions
    """
    playlists = data.get_playlists_df()
    target_playlist = pd.DataFrame({'playlist_id':data.get_target_playlists()})

    counts = playlists.merge(target_playlist).groupby('playlist_id').size().reset_index(name='interactions')
    
    # plot counts for each playlist
    #counts.plot(x='playlist_id', y='interactions', kind='scatter', figsize=(200,100))
    
    hist = counts.groupby('interactions').size().reset_index(name='counts')
    hist.plot(x='interactions', y='counts', kind='bar', fontsize=7, figsize=(150,100))

    # plot histogram
    plt.show(block=True)


def cluster_users_by_top_pop_count(clip_perc, top_n=100, only_target=True):
    """
    Return the ids of the playlists containing at least the specified percentage of top
    popular track (in descending order based on contained top pop tracks count)
    
    Parameters
    ----------
    clip_perc: (float) returns only playlist with a percentage of top pop tracks over the total 
                tracks count >= clip_perc
    top_n: consider only the most popular tracks (it should be set equal to the max
            track count among all playlists)
    only_target: (bool) consider only the target playlist

    Returns
    -------
    List of playlist_id
    """
    playlists_df = data.get_playlists_df()
    #tot_interactions = playlists_df.shape[0]
    if only_target:
        # filter only target playlist
        target_playlist_df = pd.DataFrame({'playlist_id' : data.get_target_playlists()})
        playlists_df = playlists_df.merge(target_playlist_df)

    # track_id | count
    toptracks_df = playlists_df.groupby('track_id').size().reset_index(name='count')
    #toptracks_df['relative_count'] = toptracks_df['count'] / tot_interactions
    toptracks_df = toptracks_df.sort_values('count', ascending=False)[0:top_n]

    # playlist_id | top_pop_count
    filtered_df = playlists_df.merge(toptracks_df)
    filtered_df = filtered_df.groupby('playlist_id').size().reset_index(name='top_pop_count')
    #filtered_df = filtered_df.sort_values('top_pop_count', ascending=False)

    # playlist_id | count | top_pop_count | perc
    playlists_count_df = playlists_df.groupby('playlist_id').size().reset_index(name='count')

    final_df = playlists_count_df.merge(filtered_df)
    final_df['perc'] = np.divide(final_df['top_pop_count'], final_df['count'])
    # filter only playlist with top pop perc >= clip_perc
    final_df = final_df[final_df['perc']>=clip_perc]
    final_df.sort_values(['perc','top_pop_count'], ascending=False, inplace=True)
    return final_df['playlist_id'].values

def histogram_of_top_pop_items(top_n, only_target=True):
    playlists_df = data.get_playlists_df()
    if only_target:
        # filter only target playlist
        target_playlist_df = pd.DataFrame({'playlist_id' : data.get_target_playlists()})
        playlists_df = playlists_df.merge(target_playlist_df)
    # track_id | count
    toptracks_df = playlists_df.groupby('track_id').size().reset_index(name='count')
    toptracks_df = toptracks_df.sort_values('count', ascending=False)[0:top_n]
    toptracks_df.plot(x='track_id', y='count', kind='bar', fontsize=6, figsize=(150,100))

    # plot histogram
    plt.show(block=True)

if __name__ == "__main__":
    #histogram_of_interactions()
    cluster_users_by_top_pop_count(0.5)
    #histogram_of_top_pop_items(120)
