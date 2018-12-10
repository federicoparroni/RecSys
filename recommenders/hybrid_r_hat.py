from recommenders.hybrid_base import Hybrid

class HybridRHat(Hybrid):

    def __init__(self, r_hat_array, normalization_mode, urm_filter_tracks):
        self.name = 'HybridRHat'
        self.matrices_array = r_hat_array
        self.normalized_matrices_array = None
        self.normalization_mode = normalization_mode

        super(HybridRHat, self).__init__(urm_filter_tracks=urm_filter_tracks)
