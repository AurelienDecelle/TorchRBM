#!/usr/bin/python3

import torch
from pathlib import Path
import numpy as np
import importlib
import argparse
from dataset import DatasetRBM

# This setting slightly improves performaces
torch.set_float32_matmul_precision("high")

# import command-line input arguments
def create_parser():
    parser = argparse.ArgumentParser(description='Train an RBM model.')
    parser.add_argument("-d", "--data",         type=Path,  required=True,          help='Filename of the dataset to be used for training the model.')
    parser.add_argument("-w", "--weights",                  default=False,          help="(Defaults to False). Whether to put a weights on the sequences based on the sequence identity with the neighbors.", action="store_true")
    parser.add_argument("-o", '--filename',     type=Path,  default='RBM.h5',       help='(Defaults to RBM.h5). Path to the file where to save the model.')
    parser.add_argument("-H", '--num_hiddens',  type=int,   default=100,            help='(Defaults to 100). Number of hidden units.')
    parser.add_argument('--n_save',             type=int,   default=50,             help='(Defaults to 50). Number of models to save during the training.')
    parser.add_argument('--training_mode',      type=str,   default='PCD',          help='(Defaults to PCD). How to perform the training.', choices=['PCD', 'CD', 'Rdm'])
    parser.add_argument('--num_updates',        type=int,   default=100000,         help='(Defaults to 100000). Number of epochs.')
    parser.add_argument('--lr',                 type=float, default=0.01,           help='(Defaults to 0.01). Learning rate.')
    parser.add_argument('--gibbs_steps',        type=int,   default=10,             help='(Defaults to 10). Number of Gibbs steps for each gradient estimation.')
    parser.add_argument('--batch_size',         type=int,   default=5000,           help='(Defaults to 5000). Minibatch size.')
    parser.add_argument('--num_chains',         type=int,   default=5000,           help='(Defaults to 5000). Number of parallel chains.')
    parser.add_argument("--alphabet",           type=str,   default="protein",      help="(Defaults to protein). Type of encoding for the sequences. Choose among ['protein', 'rna', 'dna'] or a user-defined string of tokens.")
    parser.add_argument('--restore',                        default=False,          help='(Defaults to False) To restore an old training.', action='store_true')
    parser.add_argument('--centered',           type=bool,  default=True,           help='(Defaults to True) Use centered gradient')
    
    parser.add_argument('--spacing',            type=str,   default='exp',          help='(Defaults to exp). Spacing to save models.', choices=['exp', 'linear'])
    parser.add_argument('--seed',               type=int,   default=0,              help='(Defaults to 0). Random seed.')
    return parser

# Select the list of training times (ages) at which saving the model.
def get_checkpoints(args):
    if args.spacing == 'exp':
        checkpoints = []
        xi = args.num_updates
        for _ in range(args.n_save):
            checkpoints.append(xi)
            xi = xi / args.num_updates**(1 / args.n_save)
        checkpoints = np.unique(np.array(checkpoints, dtype=np.int32))
    elif args.spacing == 'linear':
        checkpoints = np.linspace(1, args.num_updates, args.n_save).astype(np.int32)
    checkpoints = np.unique(np.append(checkpoints, args.num_updates))
    return checkpoints

if __name__ == '__main__':
        parser = create_parser()
        args = parser.parse_args()
        checkpoints = get_checkpoints(args)
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"Found device: {device}")
        
        training_dataset = DatasetRBM(
            path_data=args.data,
            alphabet=args.alphabet,
            compute_weights=args.weights,
            device=device
        )
        
        if training_dataset.is_binary:
            Rbm = importlib.import_module("rbm.binary.train")
        else:
            Rbm = importlib.import_module("rbm.potts.train")
            
        if args.batch_size > training_dataset.__len__():
            print(f"Warning: batch_size ({args.batch_size}) is bigger than the size of the training set ({training_dataset.__len__()}). Setting batch_size to {training_dataset.__len__()}.")
            args.batch_size = training_dataset.__len__()
            
        if not args.restore:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            Rbm.train(
                filename=args.filename,
                dataset=training_dataset,
                num_updates=args.num_updates,
                num_hiddens=args.num_hiddens,
                training_mode=args.training_mode,
                num_chains=args.num_chains,
                lr=args.lr,
                batch_size=args.batch_size,
                gibbs_steps=args.gibbs_steps,
                centered=args.centered,
                checkpoints=checkpoints,
                device=device
            )
        else:
            Rbm.restore_training(
                filename=args.filename,
                dataset=training_dataset,
                num_updates=args.num_updates,
                centered=args.centered,
                checkpoints=checkpoints,
                device=device
            )
