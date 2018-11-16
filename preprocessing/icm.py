# creating icm removing the duration feature and save to a file

# icm = m.create_icm(d.tracks_df.filter(items=['track_id', 'album_id', 'artist_id']))
# sp_icm = csr_matrix(icm)
# save_npz('../raw_data/matrices/sp_icm', sp_icm)

def create_icm(df):
    icm = np.zeros((d.N_TRACKS, d.N_ARTISTS + d.N_ALBUMS))
    for i in range(df.shape[0]):
        icm[df.iloc[i, 0], df.iloc[i, 1]] = 1
        icm[df.iloc[i, 0], df.iloc[i, 2]] = 1
    return icm