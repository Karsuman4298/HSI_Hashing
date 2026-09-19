# Table II: CLS versus all features

Run training, binary-code evaluation, and table export together from this repository:

```bash
python3 run_table2.py --device cuda
```

This launches **2,376 experiments**: 11 backbones × 9 losses × 4 HSI datasets ×
3 hash lengths (16/32/64) × 2 feature variants. Every model has both variants.
The runner updates CSV, Markdown and LaTeX after each completed experiment.
Run the identical command again to skip completed runs. An interrupted experiment
restarts from epoch one; this is experiment-level, not epoch-level, resume.

The default dataset directories match `run_comprehensive_study.py`. Override them
on your server if needed:

```bash
python3 run_table2.py --device cuda \
  --data houston2013=/path/to/houston13 \
  --data houston2018=/path/to/Houston18 \
  --data trento=/path/to/Trento \
  --data nilifossae=/path/to/NiliFossae
```

Each directory contains prepatched `HSI_Tr.mat`, `HSI_Te.mat`, `TrLabel.mat`,
`TeLabel.mat` (the existing loader also supports dataset-prefixed filenames and
bundled labels). Data patches have shape `[N,H,W,bands]`. The actual patch size
is read from the files; `--patch` does not resize prepatched inputs. Default PCA
is 30 channels, training is 100 epochs, batch size 64, learning rate 0.001, seed
345, and 10% of the test patches are query items. Dataset files are not downloaded.

Use the project's existing PyTorch/scipy/numpy/scikit-learn/einops/h5py/seaborn/
matplotlib environment. Mamba and MoE-Mamba require the official `mamba_ssm`
package and a compatible Linux/CUDA environment. No replacement Mamba is used.

Preview every command without training:

```bash
python3 run_table2.py --dry-run
```

Try a small real-data sweep first (use a separate output directory):

```bash
python3 run_table2.py --datasets trento --models ssftt --losses csq \
  --bits 16 --epochs 1 --device cuda --output_dir table2/smoke
```

## Feature definitions and reference

The layout follows Table 3 of
[Vision Transformer Hashing for Image Retrieval](https://github.com/shivram1987/VisionTransformerHashing):
paired `(cls)` and `(all)` backbone rows, loss groups across columns, 16/32/64-bit
subcolumns, and a panel for each HSI dataset. The reference's
`TransformerModel/modeling_cls.py` selects `x[:, 0]`; `modeling.py` flattens the
encoded token sequence. We use that CLS-versus-concatenation distinction.
The HSI experiments use mAP@All rather than the RGB datasets' dataset-specific
retrieval cutoffs in the reference. No reference-paper numbers are copied.

- **SSFTT:** final native CLS token versus concatenated CLS and all learned tokens.
- **SpectralFormer:** final native CLS token versus CLS and all spectral tokens.
- **MorphFormer:** final normalized native CLS token versus CLS and all morphology tokens.
- **Mamba, MoE-Mamba, SSRN, A2S2KResNet, ContextualNet, CNN-2D, CNN-3D, HybridSN:**
  these backbones have no native CLS token. Both variants extract their final
  unpooled feature map, convert each spatial/spectral position into a token,
  project channels to width 64, prepend a learned CLS token, add learned position
  embeddings, and apply one transformer encoder block (4 heads, FFN width 128,
  dropout 0.1) followed by LayerNorm. CLS hashes the resulting first token;
  all hashes the concatenation of every resulting token, including CLS.
  Their common hash-head form is dropout 0.5 → linear 1024 → ReLU → linear bits.
  The native pooling/classification heads are bypassed and removed.

The last eight are **new backbone-plus-token-adapter variants**, not unchanged
Table I architectures. The all-token head has more input weights than CLS.
Report these definitions in the manuscript; do not relabel native pooled results
as CLS or reuse Table I scores for these new variants. Native model behavior is
still available through the trainer's default `--feature_mode native`.

## Evaluation protocol

Both modes use identical prepatched train/test data, PCA fitted on training patches
only, and identical query/database indices and batch shuffle seeds. The final
training epoch is evaluated; query mAP is not used to select checkpoints. The
nine loss implementations are this project's implementations. The runner includes
learnable loss parameters (CSQ centers and GreedyHash classifier) in the optimizer.
This differs from the older trainer's model-only optimizer.

Binary codes use -1/+1 (zero logits become +1). Retrieval ranks the full database
by Hamming distance, breaking ties by saved database order. AP is the mean of
precision at each relevant rank; a query with no relevant item receives zero.
Mean AP is reported as a percentage. The saved original test indices must be
unique and query/database-disjoint; their labels/order are checked across runs.
The quality/spatial separation of the supplied prepatched train/test split remains
that of the input dataset. Upstream PCA already applied to MAT files cannot be undone.

Both dataset files and experiment configuration are recorded. Resume refuses to
mix changed settings, source code, or dataset file size/timestamp metadata in one
output directory. Use a new `--output_dir` for a different experiment protocol.
Only one runner should write to a given output directory at a time.

## Outputs

Under `table2/runs/` (or `--output_dir`):

- `config.json`: exact sweep configuration, source digest and data metadata.
- `runs/DATASET/MODEL/LOSS/BITS/MODE/`: training log, command, model weights,
  report, binary codes with sample IDs, and completion marker.
- `experiments.csv`: measured scores with feature definitions and provenance.
- `output/table2.md`, `table2.tex`, `table2_results.csv`, `missing_measurements.txt`.
- `output/table2_standalone.tex`: A3 landscape wrapper for the 28-column table.

Bold compares the two modes **within each backbone/loss/bit pair**, matching the
reference. Both values are bold for ties at two decimal places. Unmeasured cells
remain `--`. A filtered sweep retains all three bit columns, so unselected bit
lengths are explicitly unmeasured.

To regenerate the table from completed results without training:

```bash
python3 generate_table2.py generate \
  --manifest table2/runs/experiments.csv --outdir table2/runs/output
cd table2/runs/output
pdflatex table2_standalone.tex
```

The checked-in `table2/output/` is an unmeasured layout preview. Running only
`generate_table2.py generate` does not train models; use `run_table2.py` for training.
Full dataset scores must be measured on your server.

## Standalone result import

`generate_table2.py init --manifest new.csv` creates a full manifest. Each row
identifies dataset, backbone, loss, bits and `features` (`cls`/`all`). Supply either
`map_pct` in [0,100] with `protocol`, `feature_definition`, and `source`, or a
`codes_file` path relative to the manifest. A code NPZ contains `query_hash`,
`database_hash`, `query_labels`, `database_labels`, `query_ids`, `database_ids`.
Codes must be -1/+1; labels can be class IDs or binary multi-hot vectors. Use
`--strict` during generation to reject an incomplete table. No missing score is
estimated or invented.
