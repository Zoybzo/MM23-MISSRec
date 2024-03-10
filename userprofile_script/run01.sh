#!/bin/bash
cd ..
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
CUDA_LAUNCH_BLOCKING=1 
python userprofile.py \
    -d ml-1m_gender \
    -p saved/MISSRec-FHCKM_mm_full-100.pth \
    -mode inductive
cd -
