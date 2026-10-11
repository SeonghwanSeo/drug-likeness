# Custom ChemPrior datasets

ChemPrior streams Parquet datasets with `id` and `smiles` string columns.
Prepare valid, canonical SMILES supported by the bundled tokenizer.
IDs are preserved for provenance; training uses only SMILES.
Tokenization happens during training, so no Arrow conversion is needed.

## Convert SMILES to Parquet

Input must be tab-separated, with SMILES in column 0 and the source ID in column 1:

```text
CCO	compound-1
c1ccccc1	compound-2
```

```bash
pip install -e '.[train,chemprior]'
python scripts/chemprior/to_parquet.py my-molecules.smi --output data/my-parquet
```

The converter accepts `.smi` and `.smi.zst`, canonicalizes with RDKit, excludes atom-mapped molecules, and writes Zstd-compressed Parquet.
It prints row, byte and file counts.
It does not split, deduplicate or extract salt-free parents.
Use a new output directory for each conversion.

Use `--no_canonical` for already canonicalized input.
Set `--smiles_column` and `--id_column` for another column order.
Run `--help` for the remaining options.

## Add split metadata

The training loader reads row counts from native Hugging Face dataset metadata.
The converter writes only Parquet files; create a `README.md` in the output directory with file mappings, column types and split counts.

For 100,000 rows in `train.parquet`, use:

```yaml
---
configs:
- config_name: default
  data_files:
  - split: train
    path: train*.parquet
dataset_info:
  features:
  - name: id
    dtype: string
  - name: smiles
    dtype: string
  splits:
  - name: train
    num_examples: 100000
    num_bytes: 0
---
```

Replace `num_examples` with the converter's reported row count.
`num_bytes: 0` is used when no uncompressed Arrow byte count is available.
Map the actual filenames if you change the converter's filename prefix.
When providing validation or test data, add each split's file mapping and row count.
Split names select existing splits; the loader does not split data itself.

## Configure training

Save this YAML as `configs/my-chemprior.yaml`:

```yaml
dataset:
  path: data/my-parquet
  train_split: train
  val_split: null
  max_length: 127
  enumerate_stereo: false
train:
  model: small
  batch_size: 64
  global_batch_size: 64
  max_epochs: 10
run:
  save_dir: results/my-chemprior
  accelerator: gpu
  devices: 1
  precision: 32-true
  save_every_n_epochs: 1
```

```bash
python scripts/chemprior/train.py configs/my-chemprior.yaml
```

For an HF repository, use its repository ID as `dataset.path` and select a subset with `dataset.name` when needed.
Optional `revision`, `data_dir` and `data_files` follow the Hugging Face loading API.
Private repositories require credentials.

Choose a global batch size that is a positive multiple of per-device batch size times device count, and provide at least one global batch of training data.
For validation, set `val_split` to your prepared validation split and provide enough rows per worker/device for a full batch.

SMILES longer than `max_length` lexical tokens are truncated during training; EOS is appended afterward.
Files retain their full SMILES.
Training and validation omit incomplete batches.
Use `train.pretrained_model` to initialize weights, or `run.resume_checkpoint` to restore training state.
Resume does not restore the streaming data cursor.
