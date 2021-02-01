# HistoricalDictionaryExpansion

Table of contents
-----------------
- [Installation and setup](#installation)

## Installation

We strongly recommend installation via Anaconda:

* Refer to [Anaconda website and follow the instructions](https://docs.anaconda.com/anaconda/install/).

* Create a new environment:

```bash
conda create -n py37_hde python=3.7
```

* Activate the environment:

```bash
conda activate py37_hde
```

* Clone source code:

```bash
git clone https://github.com/Living-with-machines/HistoricalDictionaryExpansion.git
```

* Install dependencies:

```bash
cd /path/to/my/HistoricalDictionaryExpansion
pip install -r requirements.txt
```

Also, we use a [spaCy](https://spacy.io/) model: [en_core_web_lg](https://spacy.io/models/en#en_core_web_lg) which can be installed:

```bash
python -m spacy download en_core_web_lg
```
