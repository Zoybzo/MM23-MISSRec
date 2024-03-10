#!/bin/bash
cd ..
export CUDA_VISIBLE_DEVICES=2
CUDA_LAUNCH_BLOCKING=1 python ddp_userprofile.py \
    -d ml-1m_p4 \
    -p saved/MISSRec-FHCKM_mm_full-100.pth \
    -mode userprofile
cd -
