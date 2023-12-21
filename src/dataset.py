from typing import Union, List, Any
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import torch

def sequence_to_numeric(string : str) -> list:
    amino_letters = 'ACDEFGHIKLMNPQRSTVWY-'
    letter_map = {l : n for l, n in zip(amino_letters, range(21))}
    n_list = []
    for l in string:
        n_list.append(letter_map[l])
    return n_list

def import_from_fasta(fasta_name : Union[str, Path]) -> List[list]:
    sequences = []
    names = []
    seq = ''
    with open(fasta_name, 'r') as f:
        first_line = f.readline()
        if not first_line.startswith('>'):
            raise RuntimeError(f"The file {fasta_name} is not in a fasta format.")
        f.seek(0)
        for line in f:
            if not line.strip():
                continue
            if line.startswith('>'):
                if seq:
                    sequences.append(seq)
                header = line[1:].strip().replace(' ', '_')
                names.append(header)
                seq = ''
            else:
                seq += line.strip()
    if seq:
        sequences.append(seq)
    
    return names, sequences

class DatasetRBM(Dataset):
    def __init__(self,
                 path_data : Union[str, Path],
                 path_clu : Union[str, Path]=None):
        """Initialize the dataset.

        Args:
            path_data (Union[str, Path]): Path to the file (plain text or fasta).
            path_clu (Union[str, Path], optional): Path to the mmseqs tsv file that contains the clustering of the dataset.
        """
        self.names = []
        self.data = []
        
        # Automatically detects if the file is in fasta format and imports the data
        with open(path_data, "r") as f:
            first_line = f.readline()
        if first_line.startswith(">"):
            names, sequences = import_from_fasta(path_data)
            self.names = np.array(names)
            self.data = np.array(list(map(sequence_to_numeric, sequences)), dtype=np.int64)
        else:
            with open(path_data, "r") as f:
                for line in f:
                    self.data.append(line.strip().split())
            self.data = np.array(self.data, dtype=np.float32)
            self.names = np.arange(len(self.data))
        num_data = len(self.data)
        
        # Computes the weights to be assigned to the data
        if path_clu is None:
            self.weights = np.ones((num_data, 1), dtype=np.float32)
        else:
            c1, c2 = [], []
            with open(path_clu, "r") as f:
                for line in f:
                    n1, n2 = line.strip().split("\t")
                    c1.append(n1)
                    c2.append(n2)
            c1 = np.array(c1)
            c2 = np.array(c2)

            _, clu_inverse, clu_counts = np.unique(c1, return_counts=True, return_inverse=True)
            seq_counts = clu_counts[clu_inverse]
            seq_weights = 1 / seq_counts
            seq2weight = {seq : w for seq, w in zip(c2, seq_weights)}
            self.weights = np.array([[seq2weight[seq]] for seq in self.names], dtype=np.float32)
        
        # Shuffle the data
        perm = np.random.permutation(num_data)
        self.data = self.data[perm]
        self.names = self.names[perm]
        if path_clu is not None:
            self.weights = self.weights[perm]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx : int) -> Any:
        visible = torch.from_numpy(self.data[idx])
        weight = torch.from_numpy(self.weights[idx])
        return (visible, weight)
    
    def get_num_visibles(self) -> int:
        return self.data.shape[1]
    
    def get_num_states(self) -> int:
        return np.max(self.data) + 1
    
    def get_effective_size(self) -> int:
        return int(self.weights.sum())
    
    def get_covariance_matrix(self, device : torch.device=torch.device("cpu"), num_data : int=None) -> torch.Tensor:
        """Returns the covariance matrix of the data. If path_clu was specified, the weighted covariance matrix is computed.

        Args:
            device (torch.device, optional): Device. Defaults to torch.device("cpu").
            num_data (int, optional): Number of data to extract for computing the covariance matrix.

        Returns:
            Tensor: Covariance matrix of the dataset.
        """
        num_states = self.get_num_states()
        num_visibles = self.get_num_visibles()
        if num_data is not None:
            idxs = torch.multinomial(input=(torch.ones(self.__len__(), device=device) / self.__len__()), num_samples=num_data, replacement=False)
            data = torch.tensor(self.data[idxs], device=device)
            weights = torch.tensor(self.weights[idxs], device=device, dtype=torch.float32)
        else:
            data = torch.tensor(self.data, device=device)
            weights = torch.tensor(self.weights, device=device, dtype=torch.float32)
        if num_states != 2:
            data_oh = torch.eye(num_states, device=device)[data].reshape(-1, num_states * num_visibles).float()
        else:
            data_oh = data
        norm_weights = weights.reshape(-1, 1) / weights.sum()
        data_mean = (data_oh * norm_weights).sum(0, keepdim=True)
        cov_matrix = ((data_oh * norm_weights).mT @ data_oh) - (data_mean.mT @ data_mean)
        return cov_matrix