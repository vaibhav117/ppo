#!/bin/bash -e
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=50GB
#SBATCH --mail-type=END
#SBATCH --gres=gpu
#SBATCH --array=1
#SBATCH --job-name=PPO_Pendulum-v0_1
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --export=ALL

bash ./run-mpi4py-singularity.bash \
	python run.py --alg PPO -- seed 1 --env Pendulum-v0 
