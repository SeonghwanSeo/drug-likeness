# Drug-likeness scoring based on unsupervised learning

<img src="assets/score_distribution.png" width=600>

This repository contains API for [Drug-likeness scoring based on unsupervised learning](https://pubs.rsc.org/en/content/articlehtml/2022/sc/d1sc05248a).
Original code is available at [SeonghwanSeo/DeepDL](https://github.com/SeonghwanSeo/DeepDL).

If you want to train the model with your own dataset, please see the section [#train-model](#train-model).

If you have any problems or need help with the code, please add an issue or contact <shwan0106@kaist.ac.kr>.

### TL;DR

```bash
# evaluate a molecule with DeepDL
>>> python scoring.py 'CC(=O)Oc1ccccc1C(=O)O'
score: 88.856

# evaluate a molecule with naive setting using another model
>>> python scoring.py 'CC12C(O)CNC13C1CC(N1)C23' --naive -m chemsci-2021
score: 41.399

# screening
>>> python screening.py data/examples/chembl_1k.smi -o out.csv --naive --cuda
```

- `88.856` is the predicted score. The higher the score, the higher the **drug-likeness**.
- For fast screening, consider using `naive` setting, which evaluates a single stereoisomer.
- Multiple models are providen; see [#model-list](#model-list) for details.

## Installation

```bash
# use pip
pip install druglikeness

# from official github (python 3.9-3.13)
git clone https://github.com/SeonghwanSeo/drug-likeness.git
cd drug-likeness
pip install -e .
```

## Python API

```python
from druglikeness.deepdl import DeepDL

# Enter the name of model (see `# Model List`) or the path of your own model.
pretrained_model_name_or_path = 'extended'

# This will download the model weights if you provide the model name.
model = DeepDL.from_pretrained(pretrained_model_name_or_path, device="cpu")

# Evaluate the molecule.
score = model.scoring(smiles='CC(=O)Oc1ccccc1C(=O)O', naive=False)

# Screen the molecules.
score_list = model.screening(smiles_list=['c1ccccc1', 'CCN'], naive=True, batch_size=64)
```

## Model List

| Model Name              | Description                                                                                 |
| :---------------------- | :------------------------------------------------------------------------------------------ |
| `extended` (default)              | **New model** trained on an updated drug database. (excluding test set: FDA-approved drugs) |
| `chemsci-2025`          | **Retrained model** of `chemsci-2021` with hyperparameter tuning.                |
| `chemsci-2021`          | **Finetuned model** from the paper (PubChem pretrained, World Drug finetuned).              |
| `chemsci-2021-pretrain` | **Pretrained model** from the paper (trained on PubChem)                                    |

If your environment is offline, you can manually download the models from [Google Drive](https://drive.google.com/drive/folders/1yMxR7HwmwH8wK1mA3wgEasOZ510Ib1-o?usp=share_link).

### Model Performance

Following shows the scoring performance (AUROC) of the models on the various test datasets.

| Model          | Mode   | FDA vs ChEMBL | FDA vs ZINC15 | FDA vs GDB17 |
| -------------- | ------ | ------------- | ------------- | ------------ |
| `extended`     | strict | 0.862         | 0.961         | 0.991        |
| `extended`     | naive  | 0.861         | 0.961         | 0.989        |
| `chemsci-2025` | strict | 0.817         | 0.941         | 0.984        |
| `chemsci-2025` | naive  | 0.817         | 0.941         | 0.982        |

## Train Model

You can finetune the model with your own dataset using the pretrained model on **PubChem** dataset.

```bash
pip install -e '.[train]'

# train with the 2.8k training set from the paper
bash ./scripts/download_data.sh
python ./scripts/finetune.py --data_path ./data/train/worlddrug_not_fda.smi

python ./scripts/finetune.py --data_path <smi_file>
```

## Evaluation

```bash
# download train/test datasets
>>> bash ./scripts/download_data.sh

# evaluate the model
>>> python ./scripts/evaluate.py --cuda

# Output
Test 1489 molecules in data/test/fda.smi
Average score: 79.40051813170444

Test 1792 molecules in data/test/investigation.smi
Average score: 68.71052120625973

Test 10000 molecules in data/test/chembl.smi
Average score: 64.11581128692627

Test 10000 molecules in data/test/zinc15.smi
Average score: 52.7382759815216

Test 10000 molecules in data/test/gdb17.smi
Average score: 39.37572152862549

AUROC
FDA vs Investigation   : 0.789
FDA vs ChEMBL          : 0.862
FDA vs ZINC15          : 0.961
FDA vs GDB17           : 0.991
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
