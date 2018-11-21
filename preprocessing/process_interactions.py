class ProcessInteractions:

    """
        pre process the dataframe associated to train.csv.
        e.g. It can be used to eliminate from the dataset the first perc of samples
    """

    def __init__(self, df):
        """
        Constructor

        @Param
        df:             dataframe associated to train.csv
        """
        self.df = df

    def process(self):
        """
        Dummy method

        @Output
        df:             the unchanged dataframe
        """
        return self.df


class MaskSequentialPlaylists(ProcessInteractions):

    def __init__(self, df, percentual, seq_playlists):
        """
        Constructor

        @Param
        df:             (panda's df) dataframe associated to train.csv
        percentual      (double)     percentual to remove at the beginning of seq playlists
        seq_playlists   (lists)      list of seq playlists from which the first percentual of tracks should be deleted

        """
        super.__init__(df)
        self.perc = percentual
        self.seq_playlists = seq_playlists

    def process(self):

        """
        For each of the sequential playlists, masks from the df the first percentual of the tracks

        @Output
        df:             the modified dataframe
        """
        # to-do: adapt
        # with open('./../../raw_data/target_playlists.csv') as p:
        #     with open('./../../raw_data/train.csv') as t:
        #         with open('./../../raw_data/masked_train.csv', 'w') as new:
        #             writer = csv.writer(new)
        #             writer.writerow(['playlist_id', 'track_id'])
        #             pr = csv.reader(p)
        #             tr = csv.reader(t)
        #             next(pr)
        #             next(tr)
        #             i = next(tr)
        #             for playlist in pr:
        #                 while i[0] != playlist[0]:
        #                     writer.writerow(i)
        #                     i = next(tr)
        #                 l = []
        #                 while i[0] == playlist[0]:
        #                     l.append(i)
        #                     i = next(tr)
        #                 bound = math.ceil(len(l) * perc)
        #                 l = l[bound:-1]
        #                 for q in l:
        #                     writer.writerow(q)
