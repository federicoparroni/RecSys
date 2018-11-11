import numpy as np
import csv
import time


class Export:

    ''' Save a np matrix into a csv file ready for submission

    in:     np-matrix (#playlist x 10)
    in:     (optional) path: where to save the csv
    in:     (optional) name: of the csv

    out:    -

    '''

    @staticmethod
    def export(np_matrix, path, name, fieldnames=['playlist_id', 'track_ids']):
        name = '{}{}'.format(name, time.strftime('%d-%m-%Y %H_%M_%S.csv'))
        filepath = '{}{}'.format(path, name) 
        with open(filepath, "w") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(fieldnames)

            result = np.apply_along_axis(get_playlist_id_and_track_ids, axis=1, arr=np_matrix.astype(int))

            for l in result:
                writer.writerow(l)
        print('> Submission file created: {}'.format(filepath))


def get_playlist_id_and_track_ids(row):
    playlist_id = row[0]
    track_ids = ' '.join(map(str, row[1:]))
    return [playlist_id, track_ids]
