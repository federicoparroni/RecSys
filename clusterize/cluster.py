"""
Functions for partition the train dataset based on different criteria
and visualize some (maybe) useful characteristics.
"""

import data.data as data
import pandas as pd
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
    counts = playlists.groupby('playlist_id').size().reset_index(name='interactions')
    
    # plot counts for each playlist
    #counts.plot(x='playlist_id', y='interactions', kind='scatter', figsize=(200,100))
    
    hist = counts.groupby('interactions').size().reset_index(name='counts')
    hist.plot(x='interactions', y='counts', kind='bar', fontsize=7, figsize=(150,100))

    # plot histogram
    plt.show(block=True)


if __name__ == "__main__":
    histogram_of_interactions()
