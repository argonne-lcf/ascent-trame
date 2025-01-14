#!/bin/bash

# Find the most recent uniform0 file
latest_file=$(ls uniform0.f* 2>/dev/null | sort -t. -k2,2n | tail -n 1)

# Check if a uniform0 file was found
if [ -z "$latest_file" ]; then
    exit 1
fi

# Make a copy of the most recent file named field.f00001
cp "$latest_file" field.f00001
