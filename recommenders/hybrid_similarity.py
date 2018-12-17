from recommenders.hybrid_base import Hybrid
import data.data as data
import utils.top_k as tk

class HybridSimilarity(Hybrid):

    def __init__(self, similarity_matrices_array, normalization_mode, urm_filter_tracks):
        self.name = 'HybridSimilarity'
        self.INVERSE = False

        if similarity_matrices_array[0].shape[0] == data.N_PLAYLISTS:
            self.INVERSE = True

        self.matrices_array = similarity_matrices_array
        #========
        #self.matrices_array[0] = (tk.apply_top_k((self.matrices_array[0]).T,200)).T
        #============
        self.normalized_matrices_array = None
        self.normalization_mode = normalization_mode

        super(HybridSimilarity, self).__init__(urm_filter_tracks=urm_filter_tracks)
