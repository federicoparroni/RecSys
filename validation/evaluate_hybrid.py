from scipy.sparse import load_npz
from evaluation.map_evaluation import evaluate_map
from inout.import_rec import Import
from recommenders.hybrid.election_methods import ElectionMethods
from inout.export_rec import Export
import numpy as np


res_1 = Import.importCsv('../submissions/als_OK.csv')
res_2 = Import.importCsv('../submissions/BM_OK.csv')
res_3 = Import.importCsv('../submissions/CB_OK.csv')

res = ElectionMethods.borda_count([res_2],
                                  [0.5])

# load URM test MAP matrix
test_urm = load_npz('../raw_data/matrices/urm_test.npz')
print('> data loaded')

map10 = evaluate_map(res, test_urm)

print('Estimated map --> {}'.format(map10))

# export
Export.export(np.array(res), path='../submissions/', name='hybrid_borda')
print('> file exported')
