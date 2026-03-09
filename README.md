# Mistral
Training a Mistral model from scratch to solve long homophonic substitution ciphers

## How to train the model
### Start one training of 12 hours
- `sbatch train.slurm`

### Start multiple training jobs dependent of eachother
- `./chain_train.sh`

### How to cancel the jobs
- `scancel --name=mistral_cipher`

## Cancel all jobs for user
- `scancel -u USERNAME`