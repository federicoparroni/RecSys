import csv
import math
from random import randint

with open('./../dataset/train.csv') as f:
    with open('./../dataset/our_train.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['playlist_id', 'track_id'])
        percentage = .2
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
