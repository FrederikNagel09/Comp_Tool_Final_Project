#!/bin/bash
#BSUB -J job_name                     
#BSUB -q gpuv100                            # Queue to submit the job to
#BSUB -W 250                              # Wall time limit (100 minutes)
#BSUB -n 8                                 # Request 8 cores
#BSUB -R "rusage[mem=4GB]"                 # 
#BSUB -R "span[hosts=1]"                   # Request all cores on the same host
#BSUB -o outputs/output_lsh.out                        # Standard output redirection
#BSUB -e outputs/output_lsh.err                        # Standard error redirection

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export BLIS_NUM_THREADS=8

# Activate the virtual environment
source ~/Comp_Tool_Final_Project/.venv/bin/activate

# Run the Python script
python ~/Comp_Tool_Final_Project/scripts/run_locality_sensitive_hashing.py
