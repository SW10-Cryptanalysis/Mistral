#!/bin/bash

# The number of 12-hour jobs to string together
NUM_JOBS=16
SCRIPT="train.slurm"

echo "Submitting job 1..."
# Submit the first job and extract only the Job ID using --parsable
JOB_ID=$(sbatch --parsable $SCRIPT)
echo "Job 1 submitted with ID: $JOB_ID"

# Loop to submit the remaining jobs
for i in $(seq 2 $NUM_JOBS); do
    echo "Submitting job $i dependent on $JOB_ID..."
    
    # --dependency=afterany:$JOB_ID means "run this after the previous job ends, even if it hits the 12-hour timeout"
    JOB_ID=$(sbatch --parsable --dependency=afterany:$JOB_ID $SCRIPT)
    
    echo "Job $i submitted with ID: $JOB_ID"
done

echo "Success! $NUM_JOBS jobs are now chained in the queue."