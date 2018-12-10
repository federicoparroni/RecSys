import numpy as np

def is_equal(m1,m2):
    data_m1 = m1.data
    indptr_m1 = m1.indptr
    indices_m1 = m1.indices

    data_m2 = m2.data
    indptr_m2 = m2.indptr
    indices_m2 = m2.indices

    first_check = np.array_equal(indptr_m1, indptr_m2)
    if not first_check:
        return False
    second_check = np.array_equal(indices_m1, indices_m2)
    if not second_check:
        return False
    third_check = np.array_equal(data_m1, data_m2)
    if not third_check:
        return False
    return True
