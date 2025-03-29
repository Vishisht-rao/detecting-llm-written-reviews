#!/bin/bash

paper_type=paper
target_llm=vicuna

for i in 20
do
    python review_wm_gcg.py \
        --results_path results/${paper_type}/${target_llm}/run${i} \
        --target_str_type random \
        --paper_type $paper_type \
        --num_iter 6000 \
        --target_llm $target_llm \
        --verbose --save_state
done

echo $SLURM_NODELIST
