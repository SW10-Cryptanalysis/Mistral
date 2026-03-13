# Mistral
Training a Mistral model from scratch to solve long homophonic substitution ciphers

## How to train the model
### Start one training of 12 hours
#### Train without spaces
- `sbatch train.slurm`

#### Train with spaces
- `sbatch train.slurm --with-spaces`

### How to cancel the jobs
- `scancel --name=mistral_cipher`

## Cancel all jobs for user
- `scancel -u USERNAME`