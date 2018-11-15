import csv
import math
import numpy as np

# takes an the original training set and order the playlists according with the
# the ordered playlists contained in new_ordered_train

def mask_ordered_playlists(perc):
    with open('./../../dataset/target_playlists.csv') as p:
        with open('./../../dataset/train.csv') as t:
            with open('./../../dataset/masked_sequential_our_train.csv', 'w') as new:
                writer = csv.writer(new)
                writer.writerow(['playlist_id', 'track_id'])
                pr = csv.reader(p)
                tr = csv.reader(t)
                next(pr)
                next(tr)
                i = next(tr)
                for playlist in pr:
                    while i[0] != playlist[0]:
                        writer.writerow(i)
                        i = next(tr)
                    l = []
                    while i[0] == playlist[0]:
                        l.append(i)
                        i = next(tr)
                    bound = math.ceil(len(l)*perc)
                    l = l[bound:-1]
                    for q in l:
                        writer.writerow(q)

mask_ordered_playlists(0.2)