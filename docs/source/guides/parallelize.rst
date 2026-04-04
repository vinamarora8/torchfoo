Distributed Training with ``@parallelize``
==========================================

The :func:`~torchfoo.distributed.parallelize` decorator is the simplest way to
run a training function across multiple GPUs. It handles process spawning,
distributed setup, and cleanup automatically — you only write the single-process
logic.

Basic Usage
-----------

Decorate your training function and call it normally from the main process:

.. code-block:: python

    import torchfoo as tfoo

    @tfoo.dist.parallelize()
    def train(cfg):
        rank = tfoo.dist.get_rank()
        world_size = tfoo.dist.get_world_size()
        device = tfoo.current_device()

        model = MyModel().to(device)
        model = tfoo.ddp.make_module_ddp(model)

        for batch in dataloader:
            ...

    if __name__ == "__main__":
        train(cfg)

By default, ``@parallelize()`` uses all available GPUs,
falling back to a single CPU process if no GPUs are found.

Controlling World Size
----------------------

Pass ``world_size`` to use a specific number of processes:

.. code-block:: python

    @tfoo.dist.parallelize(world_size=2)
    def train(cfg):
        ...

If ``world_size`` exceeds the number of available GPUs, the ``gloo`` backend is
used automatically and all processes run on CPU.

Querying Distributed State
--------------------------

Inside the decorated function, use :func:`~torchfoo.distributed.get_rank` and
:func:`~torchfoo.distributed.get_world_size` to query the current process:

.. code-block:: python

    @tfoo.dist.parallelize()
    def train(cfg):
        rank = tfoo.dist.get_rank()          # 0 .. world_size-1
        world_size = tfoo.dist.get_world_size()

        if rank == 0:
            print(f"Training on {world_size} GPUs")

For rank-0-only side effects (logging, saving checkpoints), use the
:func:`~torchfoo.distributed.rank_zero_only` decorator:

.. code-block:: python

    @tfoo.dist.rank_zero_only
    def save_checkpoint(model, path):
        torch.save(model.state_dict(), path)

Wrapping a Model with DDP
--------------------------

Use :func:`~torchfoo.ddp.make_module_ddp` to wrap your model. It converts
``BatchNorm`` layers to ``SyncBatchNorm`` and wraps with
``DistributedDataParallel`` automatically. It is a no-op outside of distributed
mode, so the same code works for both single- and multi-GPU runs:

.. code-block:: python

    @tfoo.dist.parallelize()
    def train(cfg):
        device = tfoo.current_device()
        model = MyModel().to(device)
        model = tfoo.ddp.make_module_ddp(model)   # no-op if world_size == 1
        ...

Full Example
------------

.. code-block:: python

    import torch
    import torchfoo as tfoo
    from dataclasses import dataclass

    @dataclass
    class Config:
        epochs: int = 10
        lr: float = 1e-3

    @tfoo.dist.parallelize()
    def train(cfg: Config):
        rank = tfoo.dist.get_rank()
        device = tfoo.current_device()

        model = torch.nn.Linear(128, 10).to(device)
        model = tfoo.ddp.make_module_ddp(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        for epoch in range(cfg.epochs):
            # ... training loop ...

            if rank == 0:
                print(f"Epoch {epoch} done")

    if __name__ == "__main__":
        train(Config())
