# Mistral
Training a Mistral model from scratch to solve long homophonic substitution ciphers

## Initialization
- `uv sync`

## How to train the model
### Start one training of 12 hours
#### Train with spaces
- `sbatch train.slurm --dataset-path 'tokenized_normal_truncated_4000'`

#### Train without spaces
- `sbatch train.slurm --without-spaces --dataset-path 'tokenized_normal_truncated_4000'`

### How to cancel the jobs
- `scancel --name=mistral_cipher`

## Cancel all jobs for user
- `scancel -u USERNAME`