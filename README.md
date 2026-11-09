# Drug-likeness scoring based on unsupervised learning

<img src="assets/score_distribution.png" width=600>

This repository provides **DeepDL**, the model from [Drug-likeness scoring based on unsupervised learning](https://pubs.rsc.org/en/content/articlehtml/2022/sc/d1sc05248a), and **ChemPrior**, a Transformer-based SMILES model.
The original DeepDL code is available at [SeonghwanSeo/DeepDL](https://github.com/SeonghwanSeo/DeepDL).

For questions, open an issue or contact <shwan0106@kaist.ac.kr>.

## Quick Start

```bash
# Score one molecule with DeepDL.
python scoring.py 'CC(=O)Oc1ccccc1C(=O)O'
# score: 88.856

# Screen a SMILES file using a single stereoisomer per molecule.
python screening.py data/examples/chembl_1k.smi -o out.csv --naive --cuda

# Use a trained ChemPrior model.
python scoring.py 'CC(=O)Oc1ccccc1C(=O)O' \
    --arch chemprior --model SeonghwanSeo/chemprior-large-chembl
```

Higher scores indicate higher drug-likeness.
By default, DeepDL uses the lowest score among enumerated stereoisomers, while ChemPrior preserves the input stereochemistry.
`--naive` evaluates only the first isomer.
DeepDL scores range from 0 to 100; ChemPrior scores range from 0 to 1.

## Installation

Python 3.9 or later is required.

```bash
pip install druglikeness

# Install the current source.
git clone https://github.com/SeonghwanSeo/drug-likeness.git
cd drug-likeness
pip install -e .

# ChemPrior inference.
pip install -e '.[chemprior]'

# Training and dataset preparation.
pip install -e '.[train,chemprior]'
```

Run the following repository commands from the repository root.

## Python API

### DeepDL

```python
from druglikeness.deepdl import DeepDL

# A model name downloads weights from Hugging Face; local model directories also work.
model = DeepDL.from_pretrained("extended", device="cpu")
score = model.scoring("CC(=O)Oc1ccccc1C(=O)O")
scores = model.screening(["c1ccccc1", "CCN"], batch_size=64, naive=True)
```

### ChemPrior

```python
from druglikeness.chemprior import ChemPrior

model = ChemPrior.from_pretrained("SeonghwanSeo/chemprior-large-chembl")
score = model.scoring("CC(=O)Oc1ccccc1C(=O)O")
scores = model.screening(["c1ccccc1", "CCN"], batch_size=64, stereo_mode="original")

# Return summed log probability without clipping or rescaling.
log_probs = model.screening(["c1ccccc1", "CCN"], normalize=False)
```

ChemPrior loads a Hugging Face model repository or a local directory containing `config.json` and `model.pth`.
Use `model.save_pretrained("my-model")` to export a model and `ChemPrior.from_pretrained("my-model")` to load it.
Legacy `.pt` files and Lightning `.ckpt` files are no longer accepted by `ChemPrior.from_pretrained`; Lightning checkpoints remain supported for training resumption through `run.resume_checkpoint`.
The tokenizer vocabulary and parent-cleaning rules are bundled with the `druglikeness` package.
`normalize=False` returns log probability including EOS and excluding padding; stereoisomer selection follows the same scoring rule.

## Model List

### DeepDL

| Model | Hugging Face repository | Description |
| --- | --- | --- |
| `extended` (default) | [deepdl-extended](https://huggingface.co/SeonghwanSeo/deepdl-extended) | Trained on an updated drug dataset, excluding the FDA-approved test set |
| `chemsci-2025` | [deepdl-chemsci-2025](https://huggingface.co/SeonghwanSeo/deepdl-chemsci-2025) | Retrained paper model with hyperparameter tuning |
| `chemsci-2021` | [deepdl-chemsci-2021](https://huggingface.co/SeonghwanSeo/deepdl-chemsci-2021) | Paper model: PubChem pretraining followed by World Drug fine-tuning |
| `pretrain` / `chemsci-2021-pretrain` | [deepdl-pretrain](https://huggingface.co/SeonghwanSeo/deepdl-pretrain) | PubChem pretrained model from the paper |

All four DeepDL models use a 4-layer GRU with hidden size 1024, a 67-character vocabulary, and 25,328,707 parameters.
`DeepDL.from_pretrained()` accepts the aliases above, full Hub repository IDs, or local directories containing `config.json` and `model.pth`.
`model.save_pretrained("my-model")` exports the architecture configuration and a weights-only PyTorch state dictionary.
Legacy checkpoint files must be exported to this directory format before loading.

### ChemPrior

ChemPrior supports three Transformer presets.
Each size has a pretrained model and a model further trained on ChEMBL37.

| Preset | Hidden size | Layers | Attention heads | Pretrain | ChEMBL |
| --- | ---: | ---: | ---: | --- | --- |
| `small` | 256 | 8 | 4 | [Model](https://huggingface.co/SeonghwanSeo/chemprior-small-pretrain) | [Model](https://huggingface.co/SeonghwanSeo/chemprior-small-chembl) |
| `medium` | 384 | 8 | 6 | [Model](https://huggingface.co/SeonghwanSeo/chemprior-medium-pretrain) | [Model](https://huggingface.co/SeonghwanSeo/chemprior-medium-chembl) |
| `large` | 512 | 12 | 8 | [Model](https://huggingface.co/SeonghwanSeo/chemprior-large-pretrain) | [Model](https://huggingface.co/SeonghwanSeo/chemprior-large-chembl) |

Pretraining uses 2,009,894,852 unique canonical isomeric SMILES merged from ZINC20 and PubChem after exact deduplication.
Large and medium were pretrained for 5 epochs; small was pretrained for 4 epochs.
The `chembl` models continue from their corresponding pretrain weights for 15 epochs on 2,499,615 ChEMBL37 salt-free parent molecules, with 138,867 validation and 138,868 test molecules.
The datasets are linked in [SeonghwanSeo/chemprior-data](https://huggingface.co/datasets/SeonghwanSeo/chemprior-data), using the `pretrain` and `chembl` subsets.
Training truncates SMILES to 127 tokens for pretraining and 255 tokens for ChEMBL, then appends EOS, without stereo enumeration.
ChEMBL splits are random, with no scaffold, similarity, or pretrain-overlap filtering.

### DeepDL Performance

AUROC on FDA-approved drugs versus each comparison dataset:

| Model | Mode | FDA vs ChEMBL | FDA vs ZINC15 | FDA vs GDB17 |
| --- | --- | ---: | ---: | ---: |
| `extended` | strict | 0.862 | 0.961 | 0.991 |
| `extended` | naive | 0.861 | 0.961 | 0.989 |
| `chemsci-2025` | strict | 0.817 | 0.941 | 0.984 |
| `chemsci-2025` | naive | 0.817 | 0.941 | 0.982 |

## Train Model

### DeepDL Fine-tuning

Fine-tune the PubChem pretrained model with your own `.smi` file:

```bash
pip install -e '.[train]'
python scripts/deepdl/train.py --data_path my-molecules.smi

# Or use the training data from the paper.
pip install gdown  # Only needed for the legacy dataset download script.
bash scripts/deepdl/download_data.sh
python scripts/deepdl/train.py --data_path data/train/worlddrug_not_fda.smi
```

### ChemPrior Training

Choose a YAML template in `configs/chemprior/`:

```bash
pip install -e '.[train,chemprior]'
python scripts/chemprior/train.py configs/chemprior/small_1_pretrain.yaml
python scripts/chemprior/train.py configs/chemprior/small_2_midtrain.yaml

# Resume an interrupted run.
python scripts/chemprior/train.py configs/chemprior/small_1_pretrain.yaml \
    --resume_checkpoint results/chemprior-small/pretrain/checkpoints/last.ckpt
```

The templates stream the [SeonghwanSeo/chemprior-data](https://huggingface.co/datasets/SeonghwanSeo/chemprior-data) repository.
Pretraining uses the merged ZINC20 + PubChem `pretrain` subset; midtraining continues on salt-free ChEMBL37 parents in the `chembl` subset.
ChEMBL has a 90%/5%/5% train/validation/test split, without similarity filtering.

Edit the three config groups for your run:

- `dataset`: data source, splits, token limit and stereo enumeration.
- `train`: model size, initial weights, batch size, epochs and learning rate.
- `run`: output directory, devices, precision and logging.

`dataset.enumerate_stereo: false` tokenizes stored SMILES directly.
Set it to `true` to sample an enumeration of unspecified stereochemistry during training, preserving assigned stereo.
`train.batch_size` is per device; `global_batch_size` must be a positive multiple of batch size times device count.
Training writes `training_config.json`, checkpoints, and a final `config.json` plus `model.pth` under `run.save_dir`.
`train.pretrained_model` accepts a model directory or Hub repository ID for weight initialization; `run.resume_checkpoint` accepts a Lightning checkpoint to restore training state.

For your own dataset, convert tab-separated SMILES and IDs to Parquet:

```bash
python scripts/chemprior/to_parquet.py my-molecules.smi --output data/my-parquet
```

Set `dataset.path` to your prepared dataset and include its split row counts in HF README metadata.
See [Custom ChemPrior datasets](docs/chemprior-data.md) for input format, metadata and config examples.

## Evaluation

Evaluate DeepDL on the paper's test datasets:

```bash
bash scripts/deepdl/download_data.sh
python scripts/deepdl/evaluate.py --cuda
```

The legacy dataset download script requires a separate `pip install gdown`; model downloads use Hugging Face and do not require it.
The script reports mean scores and FDA AUROC against Investigation, ChEMBL, ZINC15 and GDB17.
Install `.[eval]` to use `--plot`.

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
