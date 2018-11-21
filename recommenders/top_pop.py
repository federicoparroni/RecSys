from run.als import Data

data = Data()

top_pop_tracks_id_series = data.unified_df.groupby('track_id', as_index=False).agg('count').sort_values(by='playlist_id', ascending=False)\
.reset_index(drop=True).iloc[:, 0].head(10)

#creates the csv file for submission
#top_pop_tracks_id_df.to_csv('top_pop_submission', index=False)

tracks = top_pop_tracks_id_series.apply(str).str.cat(sep=' ')
data.target_playlists_df['track_ids'] = tracks
data.target_playlists_df.to_csv('top_pop_submission', index=False)



#number of tracks in total in the raw_data
NUM_TOT_TRACKS = 20634+1

