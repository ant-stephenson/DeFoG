ENV_NAME=DeFoG
ENV_DIR=/user/work/ll20823/.conda/$ENV_NAME
ENV_FILE=/user/work/ll20823/gfscripts/$ENV_NAME/environment.yaml

# in case it already exists, start again
rm -r $ENV_DIR

source $WORK/miniforge3/bin/activate

# ensure everything install where we want
conda config --add envs_dirs $ENV_DIR
conda config --add pkgs_dirs $ENV_DIR

# create env with env and python version
conda env create -f $ENV_FILE --prefix $ENV_DIR

# test
source activate
conda activate $ENV_DIR
python --version
conda list
python -c "import torch; print(torch.__version__, flush=True)"

conda deactivate
source deactivate

# rm -r $ENV_DIR