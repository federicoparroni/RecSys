from scipy.sparse import load_npz
from evaluation.map_evaluation import evaluate_map
from helpers.manage_dataset.import_csv import Import
from election_methods import ElectionMethods
from helpers.manage_dataset.export import Export
import numpy as np

bm25_res_1 = Import.importCsv('../Hace/submissions/collaborative_BM25006315-11-2018 21_27_40.csv')
bm25_res_2 = Import.importCsv('../Hace/submissions/CB_003415-11-2018 22_19_29.csv')

res = ElectionMethods.borda_count([bm25_res_1, bm25_res_2],
                                  [0.5, 0.5])

# load URM test MAP matrix
test_urm = load_npz('../dataset/saved_matrices/sp_urm_MAP_test.npz')
print('> data loaded')

map10 = evaluate_map(res, test_urm)

print('Estimated map --> {}'.format(map10))

# export
Export.export(np.array(res), path='../submissions/', name='hybrid_borda')
print('> file exported')
