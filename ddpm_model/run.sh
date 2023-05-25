#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be ommitted.
#SBATCH -p gpu
#SBATCH --job-name=base_ddpm
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=4
#We expect that our program should not run longer than 1 day
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=6-24:00:00
#SBATCH -p gpu --gres=gpu:a100
# From here on, we can start our program

python3 train.py
