from data import Data
from matrix import M
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from scipy.sparse import load_npz
import pandas as pd
#===============================================

''' pieces of code used for save same matrices

can be done with some static methods....

'''

# d = Data()
m = M()


# code that have been used for create the URM create a sparse representation of it and save it to a file

# urm = m.create_urm(pd.read_csv('../../dataset/our_train_random.csv'))
# sp_urm = csr_matrix(urm)
# save_npz('../../dataset/saved_matrices/sp_urm_MAP_train', sp_urm)

# code for create the test matrix for evaluate the test
sp_urm = load_npz('../../dataset/saved_matrices/sp_urm.npz')
sp_urm_train = load_npz('../../dataset/saved_matrices/sp_urm_MAP_train.npz')
sp_urm_MAP = sp_urm - sp_urm_train
save_npz('../../dataset/saved_matrices/sp_urm_MAP_test', sp_urm_MAP)


#creating icm removing the duration feature and save to a file

#icm = m.create_icm(d.tracks_df.filter(items=['track_id', 'album_id', 'artist_id']))
#sp_icm = csr_matrix(icm)
#save_npz('../dataset/saved_matrices/sp_icm', sp_icm)

