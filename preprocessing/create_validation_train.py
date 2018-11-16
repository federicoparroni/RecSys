import csv
import math
from random import randint

"""
    create a validation set from the training set, by randomly picking 
    songs for the all the playlists in the training set
"""
def create_val_train_random(perc):
    with open('./../raw_data/masked_sequential_train.csv') as f:
        with open('./../raw_data/masked_sequential_our_train.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['playlist_id', 'track_id'])
            percentage = perc
            i = 0
            list = []

            spamreader = csv.reader(f)
            next(spamreader)
            for row in spamreader:
                if int(row[0]) == i:
                    list.append(row)
                else:
                    toRemove = int(math.floor(percentage * len(list)))

                    for r in range(0, toRemove):
                        list.pop(randint(0, len(list)-1))
                    for k in list:
                        writer.writerow(k)

                    i += 1
                    list = []
                    list.append(row)

"""
    create a validation set from the training set, by randomly picking songs for
     the the non sequential playlists and just the last ones for the sequentials
"""
def create_val_train_random(perc):
    with open('./../raw_data/train.csv') as f:
        with open('./../raw_data/our_train_seq_random.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['playlist_id', 'track_id'])
            i = 0
            l = []

            spamreader = csv.reader(f)
            next(spamreader)
            for row in spamreader:
                if int(row[0]) != i:
                    bound = math.ceil(len(l) * (1-perc))
                    l = l[0:bound]
                    for q in l:
                        writer.writerow(q)

                    i = int(row[0])
                    l = []
                l.append(row)

create_val_train_random(0.2)