#!/bin/bash
# SLURM script to be runned through `sbatch job.sh`
# In the following slurm options, customize (if needed) only the ones with comments

#SBATCH --job-name="SR_DARTS"            #the job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1            # number of threads
#SBATCH --time=72:00:00              # walltime limit
#SBATCH --gpus=1                     # num gpus. If set to 0 change the partition to defq or compute
#SBATCH --partition=long_gpu             # [gpu, defq, compute, debug_gpu, long_gpu, medium_gpu]
#SBATCH --account=pittorino
#SBATCH --mail-type=NONE              #notify for NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=fabrizio.pittorino@unibocconi.it
#SBATCH --output=out/%x_%j.out       # where to write standard output. %j gives job id and %x gives job name
#SBATCH --error=err/%x_%j.err        # where to write standard error.
#### #SBATCH --mem-per-cpu=8000M     # memory per cpu core, default 8GB

# PARTITIONS
# If you have cpu job change partition to compute.
#partition_name / max job duration
#long_gpu 3 days 
#gpu 1 day
#medium_gpu 3 hours
#debug_gpu 15 minutes 

export PATH="/home/Pittorino/miniconda3/bin:$PATH"
#export PATH="/home/Pittorino/miniconda3:$PATH"

module load cuda/12.3
conda init
conda activate timefs

#bash scripts/darts-search.sh
#bash scripts/train_neighbors.sh
bash scripts/darts-nasbench-search.sh
