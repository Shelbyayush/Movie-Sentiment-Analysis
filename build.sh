#!/bin/bash

# exit on error
set -o errexit

pip install -r requirements.txt

# Install Git LFS
apt-get update
apt-get install -y git-lfs
git lfs install

# Pull the LFS files
git lfs pull