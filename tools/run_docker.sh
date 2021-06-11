#!/bin/bash

# Script assumes that the built docker image has the name frtn70_env
image="frtn70_env"

# Get full path of the parent directory to this script.
parent_directory=$(dirname `dirname "$(realpath $0)"`)
echo $parent_directory

docker run -it -v "$parent_directory:/repo" $image bash
