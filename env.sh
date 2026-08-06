source "/opt/home/ohara/PARC2026_pre/venv/bin/activate"
export PYTHONPATH="/opt/home/ohara/PARC2026_pre/LIBERO-plus:/opt/home/ohara/PARC2026_pre:/opt/home/ohara/PARC2026_pre/compe"
export LIBERO_ROOT="/opt/home/ohara/PARC2026_pre/LIBERO-plus"


# ImageMagick installed in the bootstrap Conda environment
export MAGICK_HOME="/opt/home/ohara/miniforge3/envs/parc-venv-bootstrap"
export LD_LIBRARY_PATH="/opt/home/ohara/miniforge3/envs/parc-venv-bootstrap/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=osmesa
