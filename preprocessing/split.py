import data as d
import math
import pandas as pd

class Split:
    """
        split the data frame coming from the process_interactions phase and splits it
        creating an associated dataset for train
        e.g. split can be done randomly ie taking apart from each playlist a random perc of tracks
    """

    def process(self, df):
        """
        Dummy method

        @Output
        df:             the unchanged dataframe
        """
        return df


class SplitRandom(Split):

    def __init__(self, percentual):
        """
        Constructor

        percentual      (double)     percentual to remove from each playlist
        """
        self.perc = percentual

    def process(self, df):
        """
        From each playlist, removes the percentual of songs specified in the constructor by randomly picking
        songs inside the playlist

        @Param
        df:             (panda's df) dataframe associated to train.csv

        @Output
        df:             the dataframe from which we have removed the picked songs
        """
        return df.drop(df.sample(frac=self.perc).index)


class SplitRandomNonSequentiasLastSequential(Split):

    def __init__(self, percentual):
        """
        Constructor

        @Param
        df:             (panda's df) dataframe associated to train.csv
        percentual      (double)     percentual to remove from each playlist
        """
        self.perc = percentual

    def process(self, df):
        """
        From each non sequential playlist, removes the percentual of songs specified in the constructor by randomly
        picking songs inside the playlist. For each sequential playlist, removes just the last songs of the playlist.
        This is done because this way the train-test splitting of kaggle is reproduced.
        See https://www.kaggle.com/c/recommender-system-2018-challenge-polimi/discussion/69325 for details

        @Param
        df:             (panda's df) dataframe associated to train.csv

        @Output
        df:             the dataframe from which we have removed the picked songs
        """

        seq_l = d.get_target_playlists()[0:5000]
        non_seq_l = list(set(d.get_all_playlists()) - set(seq_l))

        seq_df = df[df.playlist_id.isin(seq_l)]
        non_seq_df = df[df.playlist_id.isin(non_seq_l)]

        seq_df_dropped = seq_df.groupby('playlist_id').apply(lambda x: x.iloc[:-math.floor(len(x)*self.perc)]).reset_index(drop=True)
        non_seq_df_dropped = non_seq_df.drop(non_seq_df.sample(frac=self.perc).index)
        return pd.concat([seq_df_dropped, non_seq_df_dropped]).sort_values(by='playlist_id', kind='mergesort')
