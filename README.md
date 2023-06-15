## User instruction

General user instruction for CLAMS apps is available at [CLAMS Apps documentation](https://apps.clams.ai/clamsapp/).

### System requirments

To install this app locally, one has to install the underlying OCR engine [`parseq`](https://github.com/baudm/parseq). 
Specifically, the version of `parseq` used by this app is at commit [`bc8d95c`](https://github.com/baudm/parseq/tree/bc8d95cda4666d32fa53daf2ea97ff712b71e7c7). 
In most cases (on a non-GPU computer with an `arm64` CPU), the following command should work:

```bash 
# set up your python environment first here
# now from the directory where you want to download the source code
git clone https://github.com/baudm/parseq.git
cd parseq
git checkout bc8d95cda4666d32fa53daf2ea97ff712b71e7c7
pip install -r requirements/core.txt -e . 
# conform it's installed 
cd ~
python -c "import strhub"
```

Please follow instructions in the `parseq` repository to install for other platforms.


### Configurable runtime parameter

See the app metadata for the configurable runtime parameters.