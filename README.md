# Drug-likeness scoring based on unsupervised learning

This repository contains API for [Drug-likeness scoring based on unsupervised learning](https://pubs.rsc.org/en/content/articlehtml/2022/sc/d1sc05248a).
Original code is available at [SeonghwanSeo/DeepDL](https://github.com/SeonghwanSeo/DeepDL).

If you want to train the model with your own dataset, please see the section [#train-model](#train-model).

If you have any problems or need help with the code, please add an issue or contact <shwan0106@kaist.ac.kr>.

### TL;DR

```bash
# evaluate a molecule
>>> python scoring.py 'CC(=O)Oc1ccccc1C(=O)O'
score: 88.856

# evaluate a molecule with naive setting using another model
>>> python scoring.py 'CC12C(O)CNC13C1CC(N1)C23' --naive -m chemsci-2022
score: 41.399

# screening
>>> python screening.py ./data/examples/chembl_1k.smi -o ./out.csv --naive --cuda
```

- `88.856` is the predicted score. The higher the score, the higher the **drug-likeness**.
- For fast screening, consider using `naive` setting, which evaluates a single stereoisomer.
- Multiple models are providen; see [#model-list](#model-list) for details.

## Installation

```bash
# from official github
git clone https://github.com/SeonghwanSeo/drug-likeness.git
cd drug-likeness
pip install -e .

# use pip to install
pip install git+https://github.com/SeonghwanSeo/drug-likeness.git
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
| `extended`              | **New model** trained on an updated drug database. (excluding test set: FDA-approved drugs) |
| `chemsci-2025`          | **Improved model** trained on the same dataset as the paper.                                |
| `chemsci-2021`          | The **finetuned model** from the paper (PubChem pretrained, World Drug finetuned).          |
| `chemsci-2021-pretrain` | The **pretrained model** from the paper (trained on PubChem)                                |

If your environment is offline, you can manually download the models from [Google Drive](https://drive.google.com/drive/folders/1yMxR7HwmwH8wK1mA3wgEasOZ510Ib1-o?usp=share_link).

## Train Model

You can finetune the model with your own dataset using the pretrained model on **PubChem** dataset.

```bash
pip install -e '.[train]'

# train with the 2.8k training set from the paper
bash ./scripts/download_data.sh
python ./scripts/finetune.py --data_path ./data/train/worlddrug_not_fda.smi

python ./scripts/finetune.py --data_path <smi_file>
```

## Experiments

```bash
# download train/test datasets
>>> bash ./scripts/download_data.sh

# evaluate the model
>>> python ./scripts/evaluate.py --naive --cuda

# Output
Test 1489 molecules in data/test/fda.smi
Average score: 79.42997788326143

Test 1792 molecules in data/test/investigation.smi
Average score: 68.78647222476346

Test 10000 molecules in data/test/chembl.smi
Average score: 64.26294252967834

Test 10000 molecules in data/test/zinc15.smi
Average score: 52.7776697883606

Test 10000 molecules in data/test/gdb17.smi
Average score: 41.600052004623414
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
