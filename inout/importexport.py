"""
Methods to:
- export recommendations list into a csv file
- import a csv file as recommendations list

"""
import numpy as np
import csv
import time
import os
import utils.dated_directory as datedir

def exportcsv(recs, path='submissions', name='', with_scores=False, check_len=10, add_time_suffix=True,
              fieldnames=['playlist_id', 'track_ids'], verbose=False):
    """
    Save a list of recommendations into a csv file ready for submission

    Parameters
    ----------
    :param recs:               list of recommendations
    :param path:               str, folder to save csv in
    :param name:               str, name of the file
    :param with_scores:        bool, whether to export scores or not in the csv
    :param check_len:          check if all rows contains the specified number of recommendations, set to -1 if skip the check
    :param add_time_suffix:    bool, whether to add or not the time stamp at the end of the file name
    :param fieldnames:         list of str, name of the fields to insert as first row in the csv
    """
    
    fname = '{}{}'.format(name, '_scores' if with_scores else '')
    filename = datedir.create_folder(rootpath=path, filename=fname, extension='csv', append_time_suffix=add_time_suffix)

    with open(filename, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(fieldnames)

        for row in recs:
            playlist_id = row[0]
            tracks_array = row[1:]
            # check correct number of recommendations
            _check_len(len(tracks_array), check_len)
            
            if with_scores:     # export including the scores
                # build list of str 'track_id:score'
                #   TO-DO: check if row[1:] are tuples with 2 elements or not
                tracks_ids_scores = ['{}:{}'.format(r,s) for r,s in tracks_array]
                # create line by joining track ids and scores with spaces
                tracks_scores_str = ' '.join(tracks_ids_scores)
                writer.writerow([playlist_id, tracks_scores_str])

            else:               # export without the scores
                # TO-DO: check if row[1:] are tuples with 2 elements or not
                track_ids_str = ' '.join(map(str, row[1:]))
                writer.writerow([playlist_id, track_ids_str])
    
    if verbose:  
        print('> Submission file created: {}'.format(filename))


def _check_len(n, check_len):
    if check_len > 0 and n != check_len:
        print('*** WARNING: exporting line with number of recommendations {} instead of {}'.format(n, check_len))


def importcsv(filename, skip_first_row=True, with_scores=False, check_len=10):
    """
    Load a csv file as list of recommendations

    Parameters
    ----------
    filename:         (str) path of the file
    skip_first_row:   (str) whether of not skip the first row (for example, to skip the header row)
    with_scores:      (bool) whether to import a csv the contains scores or not
    check_len:        check if all rows contains the specified number of recommendations

    Returns
    -------
    float
        list
            List of (user_id, recommendations), where recommendation
            is a list of length N of (itemid, score) tuples (if with_scores=True):
                [   (7,  [(18,0.7), (11,0.6), ...] ),
                    (13, [(65,0.9), (83,0.4), ...] ),
                    (25, [(30,0.8), (49,0.3), ...] ), ... ]
    """
    result = []
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file)
        j = 0
        for row in reader:
            if skip_first_row and j != 0:
                playlist_id = int(row[0])
                tracks_array = row[1].split(' ')
                # check correct number of recommendations
                _check_len(len(tracks_array), check_len)

                if with_scores:
                    r = []
                    for id_score in tracks_array:
                        curr_id_score_pair = id_score.split(':')
                        r.append( (int(curr_id_score_pair[0]),float(curr_id_score_pair[1])) )
                    result.append( [playlist_id] + r )
                else:
                    r = [playlist_id] + list(map(int,tracks_array)) # cast from str to int
                    result.append(r)
            else:
                j+=1
    return result
