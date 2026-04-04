import random

import torch
import pytest

import torchfoo


class TestSeedEverything:
    def test_seeds_random(self):
        torchfoo.seed_everything(42)
        a = random.random()
        torchfoo.seed_everything(42)
        b = random.random()
        assert a == b

    def test_seeds_torch(self):
        torchfoo.seed_everything(42)
        a = torch.rand(5)
        torchfoo.seed_everything(42)
        b = torch.rand(5)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        torchfoo.seed_everything(1)
        a = torch.rand(5)
        torchfoo.seed_everything(2)
        b = torch.rand(5)
        assert not torch.equal(a, b)

    def test_seeds_numpy(self):
        np = pytest.importorskip("numpy")
        torchfoo.seed_everything(42)
        a = np.random.rand(5)
        torchfoo.seed_everything(42)
        b = np.random.rand(5)
        assert (a == b).all()

    def test_deterministic_flag(self):
        torchfoo.seed_everything(42, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.benchmark is False
        torch.use_deterministic_algorithms(False)  # reset after test

    def test_default_not_deterministic(self):
        torch.use_deterministic_algorithms(False)
        torchfoo.seed_everything(42)
        assert not torch.are_deterministic_algorithms_enabled()
