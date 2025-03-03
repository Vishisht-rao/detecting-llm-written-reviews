#!/bin/bash

paper_type=prc_abstract
target_llm=vicuna

for i in 16
do
    python review_wm_gcg.py \
        --results_path results/${paper_type}/${target_llm}/run${i} \
        --target_str_type random \
        --paper_type $paper_type \
        --num_iter 2000 \
        --target_llm $target_llm \
        --verbose --save_state
done