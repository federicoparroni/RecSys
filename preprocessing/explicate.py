import pandas as pd
import numpy as np
import data.data as d
from pandas import Series

class ExplicateBase:

    """
        take the df coming after preprocess_interactions and set the ratings for all the tracks to 1
        ie maintain it implicit 
    """

    def process(self, df):
        """
        @Output
        df:             the same df with a third column for all the tracks set to 1   
        """
        df['rating'] = Series(np.ones(len(df.index)), index=df.index)
        return df

class ExplicateLinearly(ExplicateBase):
    
    """
        take the df coming after preprocess_interactions and set the ratings for all 
        the tracks in the playlist from a starting value rising up to an ending value linearly
    """
    def __init__(self, start, end):
        """
        @Param
        start:          starting value
        end:            endng value
        """
        self.start = start
        self.end = end

    def func(self, x):
        x['rating'] = np.linspace(self.start, self.end, num=len(x))
        return x

    def process(self, df):
        """
        @Output
        df:             the same df with a third column for all the tracks
        """
        return df.groupby('playlist_id').apply(self.func)