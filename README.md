# Context-based A/B test validation

`history_diagnostics` is a pure-Python3 package to validate A/B tests.

A tutorial is available [here](http://nbviewer.jupyter.org/github/sevenval/history-diagnostics/blob/development/tutorial.ipynb).

## Installation

### Requirements

Only `numpy` and `scipy` are required. `pandas` is recommended to ease up I/O.

The versions shipped by/for your OS should suffice.


### Install `history_diagnostics`

The quick and dirty way is to check out the repository
```
git clone https://github.com/sevenval/history-diagnostics
cd history-diagnostics
python3
>>> from history_diagnostics import targetspace
>>> # Do your work here
```

Optionally, you can install it, e.g. into a virtual environment
```
virtualenv -p python3 history-diagnostics-env
cd MyEnv
source bin/activate
git clone https://github.com/sevenval/history-diagnostics
cd history-diagnostics
pip install -r requirement.txt
python3 setup.py install
```

Now you can get started with the [tutorial](http://nbviewer.jupyter.org/github/sevenval/history-diagnostics/blob/development/tutorial.ipynb).
