class Split:
    """
        split the data frame coming from the process_interactions phase and splits it
        creating an associated dataset for train
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

        @Param
        df:             (panda's df) dataframe associated to train.csv
        percentual      (double)     percentual to remove from each playlist
        """
        self.perc = percentual

    def process(self, df):
        """
        From each playlist, removes the percentual of songs specified in the constructor by randomly picking
        songs inside the playlist

        @Output
        df:             the dataframe from which we have removed the picked songs
        """
        return df.drop(df.sample(frac=self.perc).index)