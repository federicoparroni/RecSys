import pandas as pd

class Data:

    def __init__(self):
        # || track_id || album_id || artist_id || duration_sec
        tracks_path = 'dataset/tracks.csv'

        # || playlist_id || track_id ||
        playlists_path = 'dataset/train.csv'

        # || playlist_id || track_id ||
        our_train_path = 'dataset/our_train.csv'

        # || playlist_id ||
        target_playlists_path = 'dataset/target_playlists.csv'

        # || all playlist_id ||
        all_playlists_path = 'dataset/all_playlist.csv'

        #creating dataframe with pandas
        self.tracks_df = pd.read_csv(tracks_path)
        self.tracks_df.columns = ['track_id', 'album_id', 'artist_id', 'duration_sec']

        self.playlists_df = pd.read_csv(playlists_path)
        self.playlists_df.columns = ['playlist_id', 'track_id']

        self.target_playlists_df = pd.read_csv(target_playlists_path)
        self.all_playlists = pd.read_csv(all_playlists_path)

        self.unified_df = self.playlists_df.merge(self.tracks_df, on='track_id')

        self.tg_pl_df_m = pd.merge(self.target_playlists_df, self.unified_df)

        self.our_train_df = pd.read_csv(our_train_path)