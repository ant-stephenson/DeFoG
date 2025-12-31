# load python
module add miniforge/20231108

# create venv
conda config --add envs_dirs /storage/hpc/08/stephe40/.conda/DeFoG
conda config --add pkgs_dirs /storage/hpc/08/stephe40/.conda/DeFoG

conda env create -f environment.yaml --prefix /storage/hpc/08/stephe40/.conda/DeFoG
# conda env update -f  environment.yaml --prune