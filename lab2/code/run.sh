#!/bin/bash

conda env create --name stat214lab2 -f environment.yaml

conda activate stat214lab2

python lab2script.py
