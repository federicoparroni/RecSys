import pandas


class TopPopModel:

    def __init__(self, train_dataframe):
        self.dataframe = train_dataframe

    def topN(self, n=10):
        counts = self.dataframe.filter(items=['track_id']).groupby(['track_id']).size().reset_index(name='counts')
        top_pop = counts.sort_values(by=['counts'], ascending=False).head(n)
        return top_pop['track_id']

    def save_submission(self, test_dataframe, filename='submissions/topPopRecommendations.csv'):
        top_pop10 = self.topN()
        print top_pop10

        tracks = top_pop10.apply(str).str.cat(sep=' ')
        test_dataframe['track_ids'] = tracks

        test_dataframe.to_csv(filename, index=False)
