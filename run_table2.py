#!/usr/bin/env python3
"""Train HSI backbone CLS/all ablations, resume completed runs, and export Table II."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from argparse import Namespace

import generate_table2 as table

ROOT = Path(__file__).resolve().parent
DATASETS = dict(zip(['houston2013', 'houston2018', 'trento', 'nilifossae'], table.DATASETS))
PATHS = {
    'houston2013': ['../houston13', 'cls_SSFTT_IP/houston13_alreadypatched_dataset'],
    'houston2018': ['../Houston18'],
    'trento': ['../Trento'],
    'nilifossae': ['../NiliFossae', 'cls_SSFTT_IP/NiliFossae_dataset'],
}
MODELS = dict(zip(['ssftt', 'mamba', 'moe_mamba', 'ssrn', 'a2s2kresnet', 'contextualnet', 'cnn2d', 'cnn3d', 'hybridsn', 'morphformer', 'spectralformer'], table.MODELS))
CLS_MODELS = {'ssftt', 'morphformer', 'spectralformer'}

LOSSES = {x.lower(): x for x in table.LOSSES}


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def resolve_data(args):
    overrides = {}
    for item in args.data:
        name, sep, value = item.partition('=')
        if not sep or name not in DATASETS or not value:
            raise ValueError('--data must be DATASET=/path/to/prepatched_data')
        overrides[name] = Path(value).expanduser().resolve()
    result = {}
    for name in args.datasets:
        candidates = [overrides[name]] if name in overrides else [ROOT / p for p in PATHS[name]]
        path = next((p.resolve() for p in candidates if p.is_dir()), candidates[0].resolve())
        if not args.dry_run:
            if not path.is_dir() or not list(path.glob('*_Tr*.mat')) or not list(path.glob('*_Te*.mat')):
                raise ValueError(f'{name}: no prepatched train/test MAT files at {path}; set --data {name}=/path')
        result[name] = str(path)
    return result


def config_for(args, paths):
    # Protect resume against silently mixing settings or changed input files.
    files = {name: {p.name: [p.stat().st_size, p.stat().st_mtime_ns]
                    for p in sorted(Path(path).glob('*.mat'))} for name, path in paths.items()}
    source = hashlib.sha256()
    for path in [Path(__file__), ROOT / 'generate_table2.py', *sorted((ROOT / 'cls_SSFTT_IP').glob('*.py'))]:
        source.update(path.name.encode())
        source.update(path.read_bytes())
    return {**{k: getattr(args, k) for k in ['datasets', 'models', 'losses', 'bits', 'epochs', 'batch_size',
             'pca', 'patch', 'num_tokens', 'seed', 'lr', 'query_ratio', 'device']},
            'data': paths, 'data_files': files, 'source_sha256': source.hexdigest(),
            'checkpoint': 'last', 'pca_fit': 'train', 'optimize_loss_parameters': True}


def build_command(args, dataset, model, loss, bits, mode, data, folder):
    return [sys.executable, '-u', str(ROOT / 'cls_SSFTT_IP/train_hsi_hashing.py'),
            '--model', model, '--device', args.device, '--dataset', dataset, '--loss_type', loss,
            '--hash_bit_length', str(bits), '--feature_mode', mode,
            '--epochs', str(args.epochs), '--batch_size', str(args.batch_size),
            '--pca', str(args.pca), '--patch', str(args.patch), '--num_tokens', str(args.num_tokens),
            '--seed', str(args.seed), '--lr', str(args.lr), '--query_ratio', str(args.query_ratio),
            '--prepatched_dir', data, '--output_dir', str(folder), '--run_name', 'model',
            '--checkpoint_selection', 'last', '--pca_fit', 'train', '--optimize_loss_parameters',
            '--binary_only', '--export_codes', str(folder / 'codes.npz')]


def export(args, rows):
    manifest = args.output_dir / 'experiments.csv'
    table.write_csv(manifest, rows)
    # Scores were computed directly from the saved codes, which remain in each run folder.
    table.generate(Namespace(manifest=manifest, outdir=args.output_dir / 'output', runner=None, strict=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    p.add_argument('--losses', nargs='+', choices=LOSSES, default=list(LOSSES))
    p.add_argument('--bits', nargs='+', type=int, choices=table.BITS, default=table.BITS)
    p.add_argument('--data', action='append', default=[], metavar='DATASET=PATH')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--pca', type=int, default=30)
    p.add_argument('--patch', type=int, default=11)
    p.add_argument('--num_tokens', type=int, default=4)
    p.add_argument('--seed', type=int, default=345)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--query_ratio', type=float, default=0.1)
    p.add_argument('--output_dir', type=Path, default=Path('table2/runs'))
    p.add_argument('--dry-run', action='store_true', help='Print commands without training or writing files')
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.num_tokens) < 1 or not 0 < args.query_ratio < 1 or args.lr <= 0:
        p.error('Use positive epochs, batch_size, num_tokens, lr and 0 < query_ratio < 1')
    for key in ['datasets', 'models', 'losses', 'bits']:
        if len(set(getattr(args, key))) != len(getattr(args, key)):
            p.error(f'Duplicate --{key} values')
    args.output_dir = args.output_dir.resolve()
    paths = resolve_data(args)
    config = config_for(args, paths)
    experiments = [(ds, model, loss, bits, mode)
                   for ds, model, loss, bits, mode in itertools.product(args.datasets, args.models, args.losses, args.bits, table.MODES)]
    print(f'{len(experiments)} training runs (CLS and all-token variants for every model).', flush=True)
    if args.dry_run:
        for ds, model, loss, bits, mode in experiments:
            folder = args.output_dir / 'runs' / ds / model / loss / str(bits) / mode
            print(shlex.join(build_command(args, ds, model, loss, bits, mode, paths[ds], folder)))
        return
    if {'mamba', 'moe_mamba'} & set(args.models):
        import importlib.util
        import torch
        if importlib.util.find_spec('mamba_ssm') is None or not torch.cuda.is_available() or args.device not in ('auto', 'cuda'):
            raise ValueError('Mamba variants require official mamba_ssm and CUDA; use a compatible server or select --models explicitly')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Settings, source code or data files changed; use a new --output_dir to avoid mixing results')
    atomic_json(config_path, config)
    rows = list(table.make_grid([DATASETS[d] for d in args.datasets], [MODELS[m] for m in args.models], [LOSSES[l] for l in args.losses]))
    # Keep all three bit columns. Unselected experiments stay explicitly unmeasured.
    for row in rows:
        row['codes_file'] = ''

    index = {tuple(row[k] for k in table.KEYS): row for row in rows}
    fingerprints = {}
    for n, (ds, model, loss, bits, mode) in enumerate(experiments, 1):
        folder = args.output_dir / 'runs' / ds / model / loss / str(bits) / mode
        folder.mkdir(parents=True, exist_ok=True)
        done = folder / 'complete.json'
        codes = folder / 'codes.npz'
        command = build_command(args, ds, model, loss, bits, mode, paths[ds], folder)
        if not done.exists():
            print(f'[{n}/{len(experiments)}] Train {ds}/{model}/{loss}/{bits}/{mode}; log: {folder / "train.log"}', flush=True)
            atomic_json(folder / 'command.json', command)
            env = dict(os.environ, MPLBACKEND='Agg')
            with (folder / 'train.log').open('w') as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                export(args, rows)
                raise SystemExit(f'Training failed ({result.returncode}); inspect {folder / "train.log"}. Rerun the same command to resume.')
        else:
            print(f'[{n}/{len(experiments)}] Resume {ds}/{model}/{loss}/{bits}/{mode}', flush=True)
        score, fingerprint = table.score_codes(codes, bits)
        if fingerprints.setdefault(ds, fingerprint) != fingerprint:
            raise ValueError(f'{ds}: retrieval split IDs/order/labels differ between runs')
        atomic_json(done, {'map_pct': score, 'split_sha256': fingerprint})
        row = index[(DATASETS[ds], MODELS[model], LOSSES[loss], str(bits), mode)]
        row.update(map_pct=f'{score:.10f}', source=str(codes),
                   feature_definition=(('Native' if model in CLS_MODELS else 'Added transformer adapter') +
                                       (' final CLS token' if mode == 'cls' else ' flattened CLS plus all feature tokens')),
                   protocol=f'prepatched; seed={args.seed}; query_ratio={args.query_ratio}; '
                            f'PCA=train; epochs={args.epochs}; checkpoint=last; '
                            f'full Hamming ranking; stable ties; zero-relevance AP=0; split={fingerprint}')
        export(args, rows)
    print(f'Table II: {args.output_dir / "output/table2.md"}', flush=True)


if __name__ == '__main__':
    main()
