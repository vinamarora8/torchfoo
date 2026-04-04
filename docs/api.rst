API Reference
=============

torchfoo
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~torchfoo.seed_everything
   ~torchfoo.current_device

torchfoo.distributed
--------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~torchfoo.distributed.is_distributed
   ~torchfoo.distributed.get_world_size
   ~torchfoo.distributed.get_rank
   ~torchfoo.distributed.rank_zero_only
   ~torchfoo.distributed.setup_distributed
   ~torchfoo.distributed.cleanup_distributed
   ~torchfoo.distributed.parallelize
   ~torchfoo.distributed.all_reduce_
   ~torchfoo.distributed.all_reduce_sum_
   ~torchfoo.distributed.all_equal
   ~torchfoo.distributed.all_concat
   ~torchfoo.distributed.all_concat_jagged

torchfoo.ddp
------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~torchfoo.ddp.make_module_ddp
