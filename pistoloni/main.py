import numpy as np
import scipy as sp
import pandas

from topPopModel import TopPopModel


train_df = pandas.read_csv('../dataset/train.csv')
test_df = pandas.read_csv('../dataset/target_playlists.csv')

tp = TopPopModel(train_df)
# top_pop10 = tp.topN()
# print top_pop10

tp.save_submission(test_df)
