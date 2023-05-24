#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be ommitted.
#SBATCH -p gpu
#SBATCH --job-name=bf512
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=4
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=14:00:00
#SBATCH -p gpu --gres=gpu:a100
# From here on, we can start our program

./finetune_barebone.sh