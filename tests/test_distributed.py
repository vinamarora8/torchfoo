import torch.multiprocessing as mp

import pytest
import torch
import torch.distributed as dist

from torchfoo.distributed.distributed import (
    is_distributed,
    get_world_size,
    get_rank,
    setup_distributed,
    cleanup_distributed,
    parallelize,
    _get_open_port,
)


@parallelize(num_gpus=2)
def _parallelize_test_fn(results_dict):
    rank = get_rank()
    results_dict[rank] = {
        "is_distributed": is_distributed(),
        "world_size": get_world_size(),
        "rank": rank,
    }


def _worker_setup_cleanup(rank, world_size, port, results):
    """Helper for multi-process tests."""
    setup_distributed(
        rank=rank,
        world_size=world_size,
        master_port=port,
    )
    results[rank] = {
        "is_distributed": is_distributed(),
        "world_size": get_world_size(),
        "rank": get_rank(),
    }
    cleanup_distributed()
    results[rank] = {**results[rank], "cleaned_up": not is_distributed()}


class TestStateQueries:
    """Tests for is_distributed, get_world_size, get_rank without a process group."""

    def test_is_distributed_false_by_default(self):
        assert not is_distributed()

    def test_world_size_1_by_default(self):
        assert get_world_size() == 1

    def test_rank_0_by_default(self):
        assert get_rank() == 0


class TestGetOpenPort:
    def test_returns_int(self):
        port = _get_open_port()
        assert isinstance(port, int)
        assert port > 0

    def test_returns_different_ports(self):
        ports = {_get_open_port() for _ in range(5)}
        # at least some should differ (not guaranteed but very likely)
        assert len(ports) >= 2


class TestSetupCleanup:
    """Tests for setup/cleanup using gloo backend on CPU (single process)."""

    def setup_method(self, method):
        """Executed before every test method."""
        import os

        os.environ.pop("MASTER_PORT", None)

    def teardown_method(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_setup_and_cleanup(self):
        port = _get_open_port()
        setup_distributed(rank=0, world_size=1, master_port=port, force=True)

        assert is_distributed()
        assert get_world_size() == 1
        assert get_rank() == 0

        cleanup_distributed()

        assert not is_distributed()

    def test_setup_skipped_when_world_size_1(self):
        port = _get_open_port()
        setup_distributed(rank=0, world_size=1, master_port=port)
        assert not is_distributed()

    def test_setup_force_with_world_size_1(self):
        port = _get_open_port()
        setup_distributed(rank=0, world_size=1, force=True, master_port=port)
        assert is_distributed()

    def test_setup_force_fails_without_port(self):
        with pytest.raises(ValueError):
            setup_distributed(rank=0, world_size=1, force=True)
        assert not is_distributed()

    def test_cleanup_noop_when_not_initialized(self):
        assert not is_distributed()
        cleanup_distributed()  # should not raise
        assert not is_distributed()

    def test_auto_backend_cpu(self):
        if torch.cuda.is_available():
            pytest.skip("test is for CPU-only environments")
        setup_distributed(rank=0, world_size=1, backend="auto", force=True)
        assert is_distributed()

    def test_master_addr_and_port(self):
        port = _get_open_port()
        setup_distributed(
            rank=0,
            world_size=1,
            master_addr="127.0.0.1",
            master_port=port,
            force=True,
        )
        assert is_distributed()


class TestMultiProcess:
    """Tests that spawn multiple processes using gloo on CPU."""

    def teardown_method(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_two_workers_setup_cleanup(self):
        port = _get_open_port()
        manager = mp.Manager()
        results = manager.dict()

        mp.spawn(
            _worker_setup_cleanup,
            args=(2, port, results),
            nprocs=2,
            join=True,
        )

        for rank in range(2):
            assert results[rank]["is_distributed"] is True
            assert results[rank]["world_size"] == 2
            assert results[rank]["rank"] == rank
            assert results[rank]["cleaned_up"] is True

    def test_two_workers_parallel(self):
        manager = mp.Manager()
        results = manager.dict()

        _parallelize_test_fn(results)

        for rank in range(2):
            assert results[rank]["is_distributed"] is True
            assert results[rank]["world_size"] == 2
            assert results[rank]["rank"] == rank


class TestSingleParallelize:
    """Tests for the parallelize decorator on CPU (single GPU = single process)."""

    def teardown_method(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_single_process_runs_function(self):
        results = []

        @parallelize(num_gpus=1)
        def fn():
            results.append(get_rank())

        fn()

        assert results == [0]

    def test_passes_args(self):
        results = []

        @parallelize(num_gpus=1, backend="auto")
        def fn(a, b):
            results.append(a + b)

        fn(2, 3)

        assert results == [5]

    def test_passes_kwargs(self):
        results = []

        @parallelize(num_gpus=1, backend="auto")
        def fn(a, b=10):
            results.append(a + b)

        fn(1, b=20)

        assert results == [21]

    def test_cleanup_after_success(self):
        @parallelize(num_gpus=1, backend="auto")
        def fn():
            pass  # world_size=1 skips distributed setup

        fn()

        assert not is_distributed()

    def test_cleanup_after_exception(self):
        @parallelize(num_gpus=1, backend="auto")
        def fn():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            fn()

        assert not is_distributed()
