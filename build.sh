#!/bin/bash

# exit on error
set -o errexit

# Install python dependencies
pip install -r requirements.txt

# Pull the LFS files (git-lfs will be pre-installed by Render)
git lfs pull