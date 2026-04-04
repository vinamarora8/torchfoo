Distributed Training Setup with ``torchfoo``
============================================

Spawning Parallel Processes
---------------------------

The :func:`~torchfoo.distributed.parallelize` decorator is the simplest way to
run a training function across multiple GPUs/CPUs. It handles process spawning,
distributed setup, and cleanup automatically — you only write the single-process
logic.

Decorate your training function and call it normally from the main process:

.. code-block:: python

    import torchfoo as tfoo

    @tfoo.dist.parallelize()
    def train(cfg):
        rank = tfoo.dist.get_rank()
        world_size = tfoo.dist.get_world_size()
        device = tfoo.current_device()

        ...

    if __name__ == "__main__":
        train(cfg)

By default, ``@parallelize()`` uses all available GPUs,
falling back to a single CPU process if no GPUs are found.

Inside the parallelized function, you can use
:func:`torchfoo.distributed.get_rank`,
:func:`torchfoo.distributed.get_world_size`,
to query the distributed state, and
:func:`torchfoo.current_device`
to get the :obj:`torch.device` corresponding to the current process.
The default :func:`torch.distributed.get_rank` etc. still work, these
ones are just nicer as they do not fail in non-distributed mode,
so the same code works for single- and multi-process settings.


Running this script does not require anything special.
Just call your train script like normal.

.. code-block:: bash

    python <script>.py


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

Extras
------

Controlling World Size with ``@parallelize``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pass ``world_size`` to use a specific number of processes:

.. code-block:: python

    @tfoo.dist.parallelize(world_size=2)
    def train(cfg):
        ...

If ``world_size`` exceeds the number of available GPUs, the ``gloo`` backend is
used automatically and all processes run on CPU.

