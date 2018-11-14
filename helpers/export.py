import numpy as np
import csv
import time


class Export:
    """
    Exposes methods to save recommendations array to a csv file
    """

    @staticmethod
    def export(recs, path, name, fieldnames=['playlist_id', 'track_ids']):
        ''' Save a np matrix of recommendations into a csv file ready for submission
        in:     recs: np-matrix (#playlist x 10) or list of:(playlist, list of:(track_id, score))
        in:     path: where to save the csv
        in:     name: of the csv

        out:    -
        '''

        np_matrix = np.array(recs)
        name = '{}{}'.format(name, time.strftime('%d-%m-%Y %H_%M_%S.csv'))
        filepath = '{}{}'.format(path, name) 
        with open(filepath, "w") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(fieldnames)

            result = np.apply_along_axis(get_playlist_id_and_track_ids, axis=1, arr=np_matrix.astype(int))

            for l in result:
                writer.writerow(l)
        print('> Submission file created: {}'.format(filepath))


    @staticmethod
    def export_with_scores(recs, path, name, fieldnames=['playlist_id', 'track_ids_and_scores']):
        ''' Save a list of recommendations and scores into a csv file
        in:     recs: list of:(playlist, list of:(track_id, score))
        in:     path: where to save the csv
        in:     name: of the csv

        out:    -
        '''
        name = '{}_scores_{}'.format(name, time.strftime('%d-%m-%Y %H_%M_%S.csv'))
        filepath = '{}{}'.format(path, name) 
        with open(filepath, "w") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(fieldnames)

            for row in recs:
                track_id = row[0]
                tracks_ids_scores = ['{}:{}'.format(r,s) for r,s in row[1]]
                line = ' '.join(tracks_ids_scores)
                
                writer.writerow([track_id, line])
        
        print('> Submission file created: {}'.format(filepath))


def get_playlist_id_and_track_ids(row):
    playlist_id = row[0]
    track_ids = ' '.join(map(str, row[1:]))
    return [playlist_id, track_ids]
