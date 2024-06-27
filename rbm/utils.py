import h5py
import numpy as np
from scipy.linalg import svd

def get_checkpoints(filename):
    checkpoints = []
    file_model = h5py.File(filename, 'r+')
    for file_key in file_model.keys():
        if "update" in file_key:
            chkp = int(file_key.split('_')[1])
            checkpoints.append(chkp)
    file_model.close()
    return np.sort(checkpoints)


def get_eigenvalues_hystory(filename):
    f = h5py.File(filename, 'r')
    updates = []
    eigenvalues = []
    for key in f.keys():
        if "update" in key:
            weight_matrix = f[key]["weight_matrix"][()]
            weight_matrix = weight_matrix.reshape(-1, weight_matrix.shape[-1])
            eig = svd(weight_matrix, compute_uv=False)
            eigenvalues.append(eig.reshape(*eig.shape, 1))
            updates.append(int(key.split("_")[1]))
    
    # Sort the results
    sorting = np.argsort(updates)
    updates = np.array(updates)[sorting]
    eigenvalues = np.array(np.hstack(eigenvalues).T)[sorting]
    f.close()
            
    return updates, eigenvalues