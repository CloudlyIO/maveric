# RADP Notebooks

## Configure Notebook environment

Use Python 3.10. The [DEVELOPER-GUIDE.md Quick Start](DEVELOPER-GUIDE.md#quick-start) installs everything,
including Jupyter. If you only want the notebooks:

```console
python3.10 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install -r notebooks/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
export PYTHONPATH="$(pwd)":$PYTHONPATH
```

This installs Jupyter Notebook 7, which has no classic nbextensions, so the old codefolding step is gone.

Optional: install ffmpeg (<https://ffmpeg.org/download.html>) for the animations in some notebooks, and point
`FFMPEG_PATH` in the repo-root `.env` at the binary:

> FFMPEG_PATH="/path_to_ffmpeg/ffmpeg"

## Prerequisites per notebook

Some notebooks need nothing else; `energy_savings.ipynb` and `load_balancing.ipynb` need the RADP Docker stack
running and a matching `.env`. See
[DEVELOPER-GUIDE.md → Notebook Prerequisites](DEVELOPER-GUIDE.md#notebook-prerequisites) for the per-notebook
table (Docker, sample data, extra setup, expected runtime).

## Sample data sets

Unpack the sample data under `notebooks/data/`:

```console
cd notebooks/data
unzip sim_data.zip              # coo_with_radp_digital_twin.ipynb
unzip mro_data.zip              # mro.ipynb
unzip energy_saving_data.zip    # energy_savings.ipynb
unzip load_balancing_data.zip   # load_balancing.ipynb
```

## Start Jupyter

From the `notebooks/` directory (paths inside the notebooks are relative to it):

```console
cd notebooks
jupyter notebook
```

This opens Jupyter in the browser. Open the notebook you want, for example `coo_with_radp_digital_twin.ipynb`, and
run it top to bottom.

## LICENSE

See [license](LICENSE)
