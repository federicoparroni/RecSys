import numpy as np
import csv


class Import:

    ''' Import a csv file as np array

    in:     filename
    in:     (optional) fieldnames, or an empty array if the first row does not contain field names

    out:    np array
    '''

    @staticmethod
    def importCsv(filename, fieldnames=['playlist_id', 'track_ids']):
        with open(filename, 'r') as csv_file:
            writer = csv.reader(csv_file)
            result = np.array([])

            j = 0
            for row in writer:
                if len(fieldnames)>0 and j != 0:
                    r = np.array([[row[0]] + row[1].split(' ')])
                    r = r.astype(np.int32)
                    print(result.shape)

                    if len(result) == 0:
                        result = r
                    else:
                        result = np.concatenate((result,r), axis=0)
                j+=1
            return result

        # print('> Submission file read: {}'.format(filename))


# Import.importCsv('submissions/10-11-2018 19_57_26_collaborative_BM25.csv')