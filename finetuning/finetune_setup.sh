#!/bin/bash

# Install dependencies
pip install git+https://github.com/huggingface/diffusers.git
pip install -U -r requirements.txt

# Save login token
python -c 'from huggingface_hub import HfFolder; HfFolder.save_token("INSERT_TOKEN_HERE")' 

# Initialize Accelerate environment
accelerate config

# Login (non-blocking if token is saved)
# huggingface-cli login
