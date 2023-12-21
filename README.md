# TorchRBM

Basic implementation in Pytorch of the Restricted Boltzmann Machine with binary hidden units.

*Example of samples obtained for a RBM trained on CelebA in black and white:*

![Alt Text](https://raw.githubusercontent.com/AurelienDecelle/TorchRBM/main/FacesBW.gif)

## Installation
Set up the environment variables by adding to the .bashrc file the following line (use your path to this repository)
```bash
export TORCHRBM=/path_to_repository/TorchRBM
```

## Training
To train a model enter:
```bash
python3 src/train.py --data <path_to_data> --filename <output_path> --model <model_type>
```
Where the input data file can be in fasta format or plain text and `model_type` should be one of the possible models:

- BernoulliBernoulliRBM
- PottsBernoulliRBM

To restore an interrupted training, enter:
```bash
python3 src/train.py --data <path_to_data> --filename <model_path> --model <model_type> --epochs <new_epochs_number> --restore
```
where `model_path` is the path to an existing RBM model and `new_epochs_number` must be larger than the previous number of training epochs.

### Training Options
All the available options can be visualized by entering
```bash
python3 src/train.py -h
```
- `-d, --data`: Path to the dataset to be used for training the model;
- `-c, --clustering`: (Optional) Path to the mmseqs `.tsv` file containing the dataset's clustering. This is used with protein sequences to mitigate the sampling bias and the effect of phylogeny that correlates data samples. Each sequence is associated with a weight inversely proportional to the number of sequences within a given sequence identity (e.g. 80%). To generate the clustering file enter:
```bash
mmseqs easy-cluster --min-seq-id 0.8 "<input_MSA>" "<prefix>" "<prefix_temp>"
```
- `-o, --filename`: (Optional, defaults to *RBM.h5*) Name of the file where to store the model;
- `--model`: (Optional, defaults to *BernoulliBernoulliRBM*) Type of model to use for the training. The possible options are: *BernoulliBernoulliRBM*, *PottsBernoulliRBM*;
- `--n_save`: (Optional, defaults to 50) Number of models to save along the training;
- `--training_mode`: (Optional, defaults to *PCD*) Training protocol. The possible options are: *CD*, *PCD*, *Rdm*;
- `--epochs`: (Optional, defaults to 100) Number of training epochs;
- `--num_hiddens`: (Optional, defaults to 100) Number of latent variables;
- `--learning_rate`: (Optional, defaults to 0.01) Learning rate;
- `--gibbs_steps`: (Optional, defaults to 10) Number of chain updates performed at each gradient estimation;
- `--batch_size`: (Optional, defaults to 1000) Minibatch size;
- `--num_chains`: (Optional, defaults to 1000) Number of Markov Chains to run in parallel;
- `--restore`: (Optional) Flag to be used for restarting from an old training;
- `--spacing`: (Optional, defaults to *exp*) Type of spacing between two checkpoints where the model is stored. The possible choices are: *linear*, *exp*;
- `--seed`: (Optional, defaults to 0) Random seed.
