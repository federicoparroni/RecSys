from preprocessing.matrix import M
import os

def create_urms(df, proc_int, split):
    m = M()

    # preprocess the interactions
    df = proc_int.process()

    # perform the split
    df_train = split.process(df)

    path = "../raw_data/new"
    os.mkdir(path)



# TODO: use df from data