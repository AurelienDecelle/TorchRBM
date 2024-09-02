# TorchRBM

Basic implementation in Pytorch of the Restricted Boltzmann Machine with binary hidden units.

*Example of samples obtained for a RBM trained on CelebA in black and white:*

![Alt Text](https://raw.githubusercontent.com/AurelienDecelle/TorchRBM/main/FacesBW.gif)

## Installation
Install the package
```bash
pip install -e .
```

## Training
To train a model enter:
```bash
python rbm/train_rbm.py --data <path_to_data> --filename <output_path>
```
The input data variables can be both in binary format or categorical (Potts).
- For the binary format, the input file is expected to be a text file in which each row represents one data point and it is a sequence of 0 or 1 separated by a space character;
- For the categorical variables, the input file must be in `fasta` format, where headers (identified by the `>` character) are alternated with sequences of symbols. For instance:

    ```
    >sequence_1
    -KLVIQAAPYGPEWLPGDADDLPL
    >sequence_2 
    -KKIILEARVNEYAPRTSNPNIPYTA
    ```

    The alphabet for the categorical variables must be specified using the `--alphabet` argument. One of the standard options `protein`, `dna` and `rna` can be chosen, or a user-defined alphabet that is coherent with the input data.

To restore an interrupted training, enter:
```bash
python rbm/train_rbm.py --data <path_to_data> --filename <model_path> --num_updates <new_number_updates> --restore
```
where `model_path` is the path to an existing RBM model and `new_number updates` must be larger than the previous number of gradient updates.

### Training Options
All the available options can be visualized by entering
```bash
python rbm/train_rbm.py -h
```
- `-d, --data`: Path to the dataset to be used for training the model;
- `-w`, `--weights`: (Optional, defaults to *False*) Whether to put weights on the sequences based on the sequence identity with the neighbors;
- `-o, --filename`: (Optional, defaults to *RBM.h5*) Name of the file where to store the model;
- `-H`, `--num_hiddens`: (Optional, defaults to 100) Number of latent variables;
- `--n_save`: (Optional, defaults to 50) Number of models to save along the training;
- `--training_mode`: (Optional, defaults to *PCD*) Training protocol. The possible options are: *CD*, *PCD*, *Rdm*;
- `--num_updates`: (Optional, defaults to 1e5) Number of gradient updates to be performed;
- `--lr`: (Optional, defaults to 0.01) Learning rate;
- `--gibbs_steps`: (Optional, defaults to 10) Number of chain updates performed at each gradient estimation;
- `--batch_size`: (Optional, defaults to 5000) Minibatch size;
- `--num_chains`: (Optional, defaults to 5000) Number of Markov Chains to run in parallel;
- `--alphabet`: (Optional, defaults to *protein*) Type of encoding for the sequences. Choose among ['protein', 'rna', 'dna'] or a user-defined string of tokens;
- `--restore`: (Optional) Flag to be used for restarting from an old training;
- `--centered`: (Optional, defaults to True) Use the centered gradient for the training. 
- `--spacing`: (Optional, defaults to *exp*) Type of spacing between two checkpoints where the model is stored. The possible choices are: *linear*, *exp*;
- `--seed`: (Optional, defaults to 0) Random seed.

## Analyze the model
In the repository `notebooks` there are two examples of how to inspect the fitted RBM model and generate new data with it.
