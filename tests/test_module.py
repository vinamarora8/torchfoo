import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import pytest

from torchfoo.distributed.distributed import (
    is_distributed,
    get_rank,
    get_world_size,
    parallelize,
)
from torchfoo.module import make_ddp
from torchfoo.utils import current_device


class TestMakeDdpNonDistributed:
    """Tests for make_ddp when not in distributed mode."""

    def test_returns_module_unchanged(self):
        m = torch.nn.Linear(4, 2)
        result = make_ddp(m)
        assert result is m

    def test_batchnorm_not_converted(self):
        m = torch.nn.Sequential(torch.nn.Linear(4, 2), torch.nn.BatchNorm1d(2))
        result = make_ddp(m)
        assert result is m
        assert isinstance(result[1], torch.nn.BatchNorm1d)
        assert not isinstance(result[1], torch.nn.SyncBatchNorm)


@parallelize(world_size=1, force=True)
def _ddp_single(results):
    dev = current_device()
    m = torch.nn.Linear(4, 2).to(dev)
    wrapped = make_ddp(m)
    results["is_ddp"] = isinstance(
        wrapped, torch.nn.parallel.DistributedDataParallel
    )


@parallelize(world_size=2)
def _ddp_two(results):
    dev = current_device()
    m = torch.nn.Linear(4, 2).to(dev)
    wrapped = make_ddp(m)
    rank = get_rank()
    results[rank] = {
        "is_ddp": isinstance(
            wrapped, torch.nn.parallel.DistributedDataParallel
        ),
        "world_size": get_world_size(),
    }


@parallelize(world_size=2)
def _ddp_batchnorm(results):
    dev = current_device()
    m = torch.nn.Sequential(
        torch.nn.Linear(4, 2), torch.nn.BatchNorm1d(2)
    ).to(dev)
    wrapped = make_ddp(m)
    rank = get_rank()
    results[rank] = {
        "has_sync_bn": isinstance(wrapped.module[1], torch.nn.SyncBatchNorm),
    }


@parallelize(world_size=2)
def _ddp_batchnorm_cpu_warning(results):
    m = torch.nn.Sequential(torch.nn.Linear(4, 2), torch.nn.BatchNorm1d(2))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        make_ddp(m)
    rank = get_rank()
    results[rank] = {
        "num_warnings": len(w),
        "warning_msg": str(w[0].message) if w else "",
    }


@parallelize(world_size=2)
def _ddp_weights_synced(results):
    dev = current_device()
    m = torch.nn.Linear(4, 2).to(dev)
    wrapped = make_ddp(m)
    rank = get_rank()
    results[rank] = {
        "weight": wrapped.module.weight.detach().cpu().tolist(),
        "bias": wrapped.module.bias.detach().cpu().tolist(),
    }


@parallelize(world_size=2)
def _ddp_preserves_parameters(results):
    dev = current_device()
    m = torch.nn.Linear(4, 2).to(dev)
    wrapped = make_ddp(m)
    rank = get_rank()
    results[rank] = {
        "same_module": wrapped.module is m,
        "weight_shape": list(wrapped.module.weight.shape),
    }


class TestMakeDdpSingleProcess:
    """Tests for make_ddp with world_size=1 and force=True."""

    def teardown_method(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_wraps_with_ddp(self):
        manager = mp.Manager()
        results = manager.dict()
        _ddp_single(results)
        assert results["is_ddp"] is True


class TestMakeDdpTwoProcesses:
    """Tests for make_ddp with world_size=2 using @parallelize."""

    def teardown_method(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_wraps_with_ddp(self):
        manager = mp.Manager()
        results = manager.dict()
        _ddp_two(results)
        for rank in range(2):
            assert results[rank]["is_ddp"] is True
            assert results[rank]["world_size"] == 2

    @pytest.mark.skipif(
        torch.cuda.device_count() < 2, reason="SyncBatchNorm requires GPU"
    )
    def test_converts_batchnorm_to_sync(self):
        manager = mp.Manager()
        results = manager.dict()
        _ddp_batchnorm(results)
        for rank in range(2):
            assert results[rank]["has_sync_bn"] is True

    def test_weights_synced_across_ranks(self):
        manager = mp.Manager()
        results = manager.dict()
        _ddp_weights_synced(results)
        assert results[0]["weight"] == results[1]["weight"]
        assert results[0]["bias"] == results[1]["bias"]

    def test_preserves_underlying_module(self):
        manager = mp.Manager()
        results = manager.dict()
        _ddp_preserves_parameters(results)
        for rank in range(2):
            assert results[rank]["same_module"] is True
            assert results[rank]["weight_shape"] == [2, 4]

    @pytest.mark.skipif(
        torch.cuda.device_count() >= 2, reason="test is for CPU/single-GPU"
    )
    def test_batchnorm_cpu_warns(self):
        manager = mp.Manager()
        results = manager.dict()
        _ddp_batchnorm_cpu_warning(results)
        for rank in range(2):
            assert results[rank]["num_warnings"] == 1
            assert "SyncBatchNorm" in results[rank]["warning_msg"]
