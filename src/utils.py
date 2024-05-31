import numpy as np
from typing import Union, Tuple
from pathlib import Path
ArrayLike = Tuple[np.ndarray, list]
import torch
from h5py import File

tokens_protein = "ACDEFGHIKLMNPQRSTVWY-"
tokens_rna = "ACGU-"
tokens_dna = "ACGT-"

def get_tokens(alphabet):
    assert isinstance(alphabet, str), "Argument 'alphabet' must be of type str"
    if alphabet == "protein":
        return tokens_protein
    elif alphabet == "rna":
        return tokens_rna
    elif alphabet == "dna":
        return tokens_dna
    else:
        return alphabet
    
def encode_sequence(sequence : str, tokens : str) -> list:
    """Takes a string sequence in input an returns the numeric encoding.

    Args:
        sequence (str): Input sequence.
        tokens (str): Alphabet to be used for the encoding.

    Returns:
        list: Encoded sequence.
    """
    letter_map = {l : n for n, l in enumerate(tokens)}
    return np.array([letter_map[l] for l in sequence])

def decode_sequence(sequence : ArrayLike, tokens : str) -> str:
    """Takes a numeric sequence in input an returns the string encoding.

    Args:
        sequence (ArrayLike): Input sequence.
        tokens (str): Alphabet to be used for the encoding.

    Returns:
        list: Decoded sequence.
    """
    return ''.join([tokens[aa] for aa in sequence])

def import_from_fasta(fasta_name : Union[str, Path]) -> Tuple[list, list]:
    """Import data from a fasta file.

    Args:
        fasta_name (Union[str, Path]): Path to the fasta file.

    Raises:
        RuntimeError: The file is not in fasta format.

    Returns:
        Tuple[list, list]: headers, sequences.
    """
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
    return np.array(names), np.array(sequences)
    
def write_fasta(fname : str, headers : ArrayLike, sequences : ArrayLike, numeric_input : bool=False, remove_gaps : bool=False, alphabet : str="protein"):
    """Generate a fasta file with the input sequences.

    Args:
        fname (str): Name of the output fasta file.
        headers (ArrayLike): List of sequences' headers.
        sequences (ArrayLike): List of sequences.
        numeric_input (bool, optional): Whether the sequences are in numeric (encoded) format or not. Defaults to False.
        remove_gaps (bool, optional): If True, removes the gap from the alignment. Defaults to False.
        alphabet (str, optional): Selects the type of sequences. Possible chooses are ("protein", "rna"). Defaults to "protein".

    Raises:
        RuntimeError: The alphabet is not a valid choice.
    """
    tokens = get_tokens(alphabet)

    if numeric_input:
        # Decode the sequences
        seqs_decoded = np.vectorize(decode_sequence, signature="(m), () -> ()")(sequences, tokens)
    else:
        seqs_decoded = sequences.copy()
    if remove_gaps:
        seqs_decoded = np.vectorize(lambda s: s.replace("-", ""))(seqs_decoded)
        
    with open(fname, 'w') as f:
        for name_seq, seq in zip(headers, seqs_decoded):
            f.write('>' + name_seq + '\n')
            f.write(seq)
            f.write('\n')
            
def compute_weights(data : ArrayLike, th : float=0.8, device : torch.device=torch.device("cpu")) -> np.ndarray:
    """Computes the weight to be assigned to each sequence 's' in 'data' as 1 / n_clust, where 'n_clust' is the number of sequences
    that have a sequence identity with 's' >= th.

    Args:
        data (ArrayLike): Encoded input dataset.
        th (float, optional): Sequence identity threshold for the clustering. Defaults to 0.8.
        device (torch.device): Device.

    Returns:
        np.ndarray: Array with the weights of the sequences.
    """
    data = torch.tensor(data, device=device)
    assert len(data.shape) == 2, "'data' must be a 2-dimensional array"
    _, L = data.shape
    
    @torch.jit.script
    def get_sequence_weight(s : torch.Tensor, data : torch.Tensor, L : int, th : float):
        seq_id = torch.sum(s == data, dim=1) / L
        n_clust = torch.sum(seq_id >= th)
        return 1. / n_clust

    weights = torch.vstack([get_sequence_weight(s, data, L, th) for s in data])
    return weights.cpu().numpy()

def validate_alphabet(sequences : ArrayLike, tokens : str):
    all_char = "".join(sequences)
    tokens_data = "".join(sorted(set(all_char)))
    sorted_tokens = "".join(sorted(tokens))
    if not sorted_tokens == tokens_data:
        raise KeyError(f"The chosen alphabet is incompatible with the Multi-Sequence Alignment. The missing tokens are: {[c for c in tokens_data if c not in sorted_tokens]}")
    
def load_params(fname : str, device : torch.device=torch.device("cpu"), epoch : int=None, set_seed : bool=False):
    """Load the parameters of a previously saved RBM model.

    Args:
        fname (str): Path to the model.
        device (torch.device, optional): Device. Defaults to torch.device("cpu").
        epoch (int, optional): Epoch to be loaded. If None, the last one is taken. Defaults to None.
        set_seed (bool, optional): Whether to set the random seed to the value of the checkpoint. Defaults to False.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Parameters of the model.
    """
    file_model = File(fname, 'r+')
    # Automatically selects the last epoch if not given
    if not epoch:
        epoch = 0
        for key in file_model.keys():
            if "epoch" in key:
                ep = int(key.split('_')[1])
                if epoch < ep:
                    epoch = ep
    key_epoch = f"epoch_{epoch}"
    
    # Set the random seed
    if set_seed:
        torch.set_rng_state(torch.tensor(np.array(file_model[key_epoch]['torch_rng_state'])))
        np_rng_state = tuple([file_model[key_epoch]['numpy_rng_arg0'][()].decode('utf-8'),
                                file_model[key_epoch]['numpy_rng_arg1'][()],
                                file_model[key_epoch]['numpy_rng_arg2'][()],
                                file_model[key_epoch]['numpy_rng_arg3'][()],
                                file_model[key_epoch]['numpy_rng_arg4'][()]])
        np.random.set_state(np_rng_state)
    
    # Load parameters
    weight_matrix = torch.tensor(file_model[key_epoch]["weight_matrix"][()], device=device)
    vbias = torch.tensor(file_model[key_epoch]["vbias"][()], device=device)
    hbias = torch.tensor(file_model[key_epoch]["hbias"][()], device=device)
    params = (vbias, hbias, weight_matrix)
    return params