import csv

# takes an the original_csv training set and order the playlists according with the
# the ordered playlists contained in new_ordered_train

with open('./../raw_data/train.csv') as f:
    with open('./../raw_data/train_sequential.csv') as csv_file:
        with open('./../raw_data/new_ordered_train.csv', 'w') as new:
            writer = csv.writer(new)
            writer.writerow(['playlist_id', 'track_id'])
            t = csv.reader(f)
            ts = csv.reader(csv_file)
            next(t)
            next(ts)
            rn = next(ts)
            arrived = False
            for ro in t:
                if int(ro[0]) == int(rn[0]) and not arrived:
                    writer.writerow(rn)
                    try:
                        rn = next(ts)
                    except:
                        arrived = True

                else:
                    writer.writerow(ro)
