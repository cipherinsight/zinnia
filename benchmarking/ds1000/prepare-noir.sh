#!/bin/bash

# Check if the case parameter is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <case>"
  exit 1
fi

# Assign the case parameter to a variable
case="$1"

# Create the directory if it doesn't exist
mkdir -p "$case"

# Create the files
touch "$case/main.nr"
touch "$case/Prover.toml"

echo "Files created in the '$case' directory."