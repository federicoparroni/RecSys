from main import Data
from main import M
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
#===============================================

''' pieces of code used for save same matrices

can be done with some static methods....

'''

d = Data()
m = M()


#code that have been used for create the URM create a sparse representation of it and save it to a file

urm = m.create_urm(d.playlists_df)
sp_urm = csr_matrix(urm)
save_npz('../dataset/saved_matrices/sp_urm', sp_urm)


#creating icm removing the duration feature and save to a file

icm = m.create_icm(d.tracks_df.filter(items=['track_id', 'album_id', 'artist_id']))
sp_icm = csr_matrix(icm)
save_npz('../dataset/saved_matrices/sp_icm', sp_icm)

