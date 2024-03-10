#!/bin/bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

python ../preprocessing/process_userprofile.py \
  --dataset ml-1m_p6 \
  --input_path /home/lllrrr/Datasets/ml-1m \
  --output_path /home/lllrrr/Datasets/ml-1m_p6 \
  --word_drop_ratio 0.5
