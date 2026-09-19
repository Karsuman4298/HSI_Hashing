"""Check real model gradients, feature dimensions, and all project loss functions."""
import importlib
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cls_SSFTT_IP'))
from TokenFeatureHashNet import TokenFeatureHashNet
import hash_losses as losses

torch.set_num_threads(1)

BACKBONES = [
    ('SSRNHashNet', dict(pca_channels=30)),
    ('A2S2KResNetHashNet', dict(pca_channels=30)),
    ('ContextualHashNet', dict(pca_channels=30)),
    ('CNN2DHashNet', dict(in_channels=30)),
    ('CNN3DHashNet', dict(input_channels=30, patch_size=11)),
    ('HybridSNHashNet', dict(in_channels=30, patch_size=11)),
]


@pytest.mark.parametrize('name,kwargs', BACKBONES)
@pytest.mark.parametrize('mode', ['cls', 'all'])
def test_adapted_backbone_gradients(name, kwargs, mode):
    x = torch.randn(2, 1, 30, 11, 11)
    native = getattr(importlib.import_module(name), name)(hash_bit_length=16, **kwargs)
    model = TokenFeatureHashNet(native, x[:1], 16, mode)
    codes, features = model(x, return_features=True)
    assert codes.shape == (2, 16)
    assert features.shape[1] == (64 if mode == 'cls' else model.position.shape[1] * 64)
    codes.square().mean().backward()
    assert model.cls_token.grad.abs().sum() > 0
    assert next(model.backbone.parameters()).grad.abs().sum() > 0
    assert torch.isfinite(codes).all()


@pytest.mark.parametrize('name,kwargs', [
    ('SSFTTHashNet', dict(pca_channels=30)),
    ('SpectralFormerHashNet', dict(in_channels=30, patch_size=11)),
    ('MorphFormerHashNet', dict(in_channels=30, patch_size=11)),
])
@pytest.mark.parametrize('mode', ['cls', 'all'])
def test_native_token_modes(name, kwargs, mode):
    kwargs = dict(kwargs)
    kwargs.update({'pool': mode} if name == 'SpectralFormerHashNet' else {'use_all_tokens': mode == 'all'})
    model = getattr(importlib.import_module(name), name)(hash_bit_length=16, **kwargs)
    codes, features = model(torch.randn(2, 1, 30, 11, 11), return_features=True)
    assert codes.shape == (2, 16)
    assert features.shape[1] == 64 if mode == 'cls' else features.shape[1] > 64
    codes.square().mean().backward()
    assert model.cls_token.grad.abs().sum() > 0
    assert torch.isfinite(codes).all()


@pytest.mark.parametrize('name', ['CSQLoss', 'DPNLoss', 'DSHLoss', 'GreedyHashLoss',
                                    'HashNetLoss', 'IDHNLoss', 'OrthoHashLoss', 'DSPCHLoss', 'BatchDHNNLoss'])
def test_losses_backward(name):
    kwargs = dict(bit_length=16, num_classes=6)
    if name in ('DSHLoss', 'HashNetLoss', 'IDHNLoss'):
        kwargs['num_train'] = 8
    criterion = getattr(losses, name)(**kwargs)
    codes = torch.randn(4, 16, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1])
    extra = {} if name in ('CSQLoss', 'DPNLoss') else dict(ind=torch.arange(4), epoch=0)
    value = criterion(codes, labels, **extra)
    value.backward()
    assert torch.isfinite(value)
    assert torch.isfinite(codes.grad).all()


@pytest.mark.parametrize('name', ['MambaHashNet', 'MoEMambaHashNet'])
@pytest.mark.parametrize('mode', ['cls', 'all'])
def test_mamba_adapters_cuda(name, mode):
    if not torch.cuda.is_available():
        pytest.skip('Official mamba_ssm requires a Linux/CUDA test environment')
    pytest.importorskip('mamba_ssm')
    x = torch.randn(2, 1, 30, 11, 11, device='cuda')
    native = getattr(importlib.import_module(name), name)(in_channels=30, hash_bit_length=16).cuda()
    model = TokenFeatureHashNet(native, x[:1], 16, mode).cuda()
    codes = model(x)
    codes.square().mean().backward()
    assert codes.shape == (2, 16)
    assert model.cls_token.grad.abs().sum() > 0


def test_default_sweep_has_both_modes_for_all_models(monkeypatch, capsys):
    import run_table2
    monkeypatch.setattr(sys, 'argv', ['run_table2.py', '--dry-run'])
    run_table2.main()
    output = capsys.readouterr().out
    commands = output.splitlines()[1:]
    assert len(commands) == 2376
    for model in run_table2.MODELS:
        for mode in ('cls', 'all'):
            assert sum(f'--model {model} ' in c and f'--feature_mode {mode} ' in c for c in commands) == 108


def test_torch_and_numpy_ranking_agree():
    import train_hsi_hashing as trainer
    from generate_table2 import retrieval_map
    q = torch.ones(2, 16)
    db = torch.ones(4, 16)
    ql, dl = torch.tensor([0, 9]), torch.tensor([0, 1, 0, 0])
    actual = trainer.calculate_mAP(q, ql, db, dl) * 100
    assert actual == pytest.approx(retrieval_map(q.numpy(), db.numpy(), ql.numpy(), dl.numpy()))
