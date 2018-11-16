import csv

"""
    from the new order of songs given just for the target playlists, creates a 
    new csv with the order of songs which is respected for the target playlists
    (specified in the new_ordered_train.csv file)
"""

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
