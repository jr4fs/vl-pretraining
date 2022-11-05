# The name of this experiment.
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lxmert
# See Readme.md for option details.
PYTHONPATH=$PYTHONPATH:./src \
    python test.py