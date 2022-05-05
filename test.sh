#!/bin/bash
#SBATCH -J test             
#SBATCH -A agasi             
#SBATCH -o test.out    
#SBATCH -p debug          
#SBATCH -N 1              
#SBATCH -n 1              
#SBATCH --gres=gpu:1            
#SBATCH --cpus-per-task=10       
#SBATCH --time=00:15:00      

module purge 
module load centos7.9/lib/cuda/11.3 

#eval "$(/truba/sw/centos7.9/lib/anaconda3/2021.11/bin/conda shell.bash hook)"
eval "$(/truba/home/agasi/miniconda3/bin/conda shell.bash hook)"
conda activate tik4 

python test.py