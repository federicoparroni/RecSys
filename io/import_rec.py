import numpy as np
import csv

class Import:

    ''' Import a csv file as np array
    in:     filename
    in:     (optional) fieldnames, or an empty array if the first row does not contain field names

    out:    np array
    '''
    @staticmethod
    def importCsv(filename, skip_first_row=True):
        with open(filename, 'r') as csv_file:
            reader = csv.reader(csv_file)
            result = np.array([])

            j = 0
            for row in reader:
                if skip_first_row and j != 0:
                    r = np.array([[row[0]] + row[1].split(' ')])
                    r = r.astype(np.int32)

                    if len(result) == 0:
                        result = r
                    else:
                        result = np.concatenate((result,r), axis=0)
                j+=1
            return result


    ''' Import a csv file as np array
    in:     filename
    in:     (optional) fieldnames, or an empty array if the first row does not contain field names

    out:    np array
    '''
    @staticmethod
    def importCsv_with_scores(filename, skip_first_row=True):
        with open(filename, 'r') as csv_file:
            reader = csv.reader(csv_file)
            result = []

            j = 0
            for row in reader:
                if skip_first_row and j != 0:
                    playlist_id = int(row[0])
                    tracks_ids_scores = row[1].split(' ')

                    r = []
                    for id_score in tracks_ids_scores:
                        curr_id_score_pair = id_score.split(':')
                        r.append((int(curr_id_score_pair[0]),float(curr_id_score_pair[1])))

                    result.append([playlist_id, r])
                else:
                    j+=1
            return result


#r=Import.importCsv_with_scores('submissions/collaborative_BM25_scores_14-11-2018 11_48_06.csv')
#print(r)