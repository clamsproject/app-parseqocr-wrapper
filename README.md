# PARSeq Scene Text Recognition Wrapper

## Description

This app wraps the [PARSeq](https://github.com/baudm/parseq) scene text recognition engine into a CLAMS app.

## User instruction

General user instructions for CLAMS apps is available at [CLAMS Apps documentation](https://apps.clams.ai/clamsapp).

### System requirments

To install this app locally, one has to install the underlying OCR engine [`parseq`](https://github.com/baudm/parseq). 
Specifically, the version of `parseq` used by this app is at commit [`bc8d95c`](https://github.com/baudm/parseq/tree/bc8d95cda4666d32fa53daf2ea97ff712b71e7c7). 
In most cases (on a non-GPU computer with an Intel or AMC (`amd64` architecture) CPU), the following command should work:

```bash 
# set up your python environment first before going into the following
# e.g. `conda create blah blah`
# now, from the directory where you want to download the source code
git clone https://github.com/baudm/parseq.git
cd parseq
git checkout bc8d95cda4666d32fa53daf2ea97ff712b71e7c7
pip install -r requirements/core.txt -e . 
# let's confirm it's installed by running the following command
## get out of the cloned parseq source directory
cd ~
## try import parseq into python 
python -c "import strhub"
```

If you see an `ImportError` from the above, the installation went wrong.
Please visit the [`parseq` repository](https://github.com/baudm/parseq) for more detailed installation instructions,
including instructions for other computing platforms such as GPU-enabled computers, ARM-based computers, etc.

### Configurable runtime parameter

For the full list of parameters, please refer to the app metadata from [CLAMS App Directory](https://apps.clams.ai/clamsapp/) or [`metadata.py`](metadata.py) file in this repository.
