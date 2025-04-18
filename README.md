# Drug-likeness scoring based on unsupervised learning

This repository contains API for [Drug-likeness scoring based on unsupervised learning](https://pubs.rsc.org/en/content/articlehtml/2022/sc/d1sc05248a).

If you want to train the model with your own dataset, please use [original repository](https://github.com/SeonghwanSeo/DeepDL).
You can still use this repository with your own trained model.

If you have any problems or need help with the code, please add an issue or contact <shwan0106@kaist.ac.kr>.

### TL;DR

```bash
# scoring
>>> python scoring.py 'c1ccccc1'
score: 83.699
>>> python scoring.py 'CCC(C)C(C)CC' -m worlddrug --naive
score: 76.283
# screening
>>> python screening.py ./test/chembl.smi -o ./out.csv --naive
```

- `83.699` is the predicted score. The higher the predicted value is, the higher druglikeness is.
- For fast screening, we can consider single stereo isomer for each molecule using naive setting.
- We provide two model: `base (default; pubchem_worddrug)` and `worlddrug`.

## Installation

```bash
# from official github
git clone https://github.com/SeonghwanSeo/drug-likeness.git
cd drug-likeness
pip install -e .

# use pip
pip install git+https://github.com/SeonghwanSeo/drug-likeness.git
```

## Python API

```python
from druglikeness import DrugLikeness

pretrained_model_name_or_path = 'base' # base | worlddrug | own weight path
model = DrugLikeness.from_pretrained(pretrained_model_name_or_path, "cpu")
score = model.evaluate(smiles='c1ccccc1', naive=False)
```

## Citation

```bibtex
@article{lee2022drug,
  title={Drug-likeness scoring based on unsupervised learning},
  author={Lee, Kyunghoon and Jang, Jinho and Seo, Seonghwan and Lim, Jaechang and Kim, Woo Youn},
  journal={Chemical science},
  volume={13},
  number={2},
  pages={554--565},
  year={2022},
  publisher={Royal Society of Chemistry}
}
```
