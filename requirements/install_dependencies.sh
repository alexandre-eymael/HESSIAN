#!/bin/bash

# Get current directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get the directory name from the input string
directory="$1"

# Make sure DIR/directory exists
if [ ! -d "$DIR/$directory" ]; then
  echo "Directory `$directory` does not exist"
  exit 1
fi

# For all directory/requirements*.txt in the directory, run pip install
for file in $DIR/$directory/requirements*.txt; do
  pip install -r $file
done


