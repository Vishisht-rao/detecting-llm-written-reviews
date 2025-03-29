#!/bin/bash

# python evaluate.py --input_dir results/paper/llama_6000 --mode gcg --llm llama
# python evaluate.py --input_dir results/prc_abstract/vicuna_6000 --mode gcg --llm vicuna
python evaluate.py --input_dir results/paper/vicuna_6000 --mode gcg --llm vicuna

echo $SLURM_NODELIST
