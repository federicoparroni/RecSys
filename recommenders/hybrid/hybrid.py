from scipy.sparse import load_npz
from evaluation.map_evaluation import evaluate_map
from helpers.import_csv import Import
from recommenders.hybrid.election_methods import ElectionMethods
from helpers.export import Export
import numpy as np

bm25_res = Import.importCsv_with_scores('submissions/collaborative_BM25_scores_14-11-2018 11_48_06.csv')

res = ElectionMethods.borda_count_on_scores([bm25_res], [1])

# load URM test MAP matrix
test_urm = load_npz('raw_data/matrices/sp_urm_test_MAP.npz')
print('> data loaded')

map10 = evaluate_map(res, test_urm)

print('Estimated map --> {}'.format(map10))

# export
Export.export(np.array(res), path='submissions/', name='hybrid_borda_on_scores')
print('> file exported')