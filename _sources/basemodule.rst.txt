.. _basemodule:

.. currentmodule:: parametricmatrixmodels.modules

..
    _comment: BaseModule is treated specially to both appear first and to have its members sectioned informatively.
    _comment: The members that MUST be overridden by subclasses are excluded from the full list and presented first.

``BaseModule``
==============
.. autoclass:: parametricmatrixmodels.modules.BaseModule
    :show-inheritance:
    :exclude-members: __init__, is_ready, get_num_trainable_floats, _get_callable, compile, get_output_shape, get_hyperparameters, get_params, set_params, name, __repr__, __call__, set_hyperparameters, get_state, set_state, set_precision, astype, serialize, deserialize, _abc_impl

Methods that must be overridden by subclasses
---------------------------------------------

.. autosummary::
    :nosignatures:

    ~BaseModule.__init__
    ~BaseModule.is_ready
    ~BaseModule._get_callable
    ~BaseModule.compile
    ~BaseModule.get_output_shape
    ~BaseModule.get_hyperparameters
    ~BaseModule.get_params
    ~BaseModule.set_params

.. automethod:: parametricmatrixmodels.modules.BaseModule.__init__
.. automethod:: parametricmatrixmodels.modules.BaseModule.is_ready
.. automethod:: parametricmatrixmodels.modules.BaseModule._get_callable
.. automethod:: parametricmatrixmodels.modules.BaseModule.compile
.. automethod:: parametricmatrixmodels.modules.BaseModule.get_output_shape
.. automethod:: parametricmatrixmodels.modules.BaseModule.get_hyperparameters
.. automethod:: parametricmatrixmodels.modules.BaseModule.get_params
.. automethod:: parametricmatrixmodels.modules.BaseModule.set_params

Methods with default implementations
------------------------------------

.. autosummary::
    :nosignatures:

    ~BaseModule.name
    ~BaseModule.__repr__
    ~BaseModule.__call__
    ~BaseModule.set_hyperparameters
    ~BaseModule.get_num_trainable_floats
    ~BaseModule.get_state
    ~BaseModule.set_state
    ~BaseModule.set_precision
    ~BaseModule.astype
    ~BaseModule.serialize
    ~BaseModule.deserialize

.. autoproperty:: parametricmatrixmodels.modules.BaseModule.name
.. automethod:: parametricmatrixmodels.modules.BaseModule.__repr__
.. automethod:: parametricmatrixmodels.modules.BaseModule.__call__
.. automethod:: parametricmatrixmodels.modules.BaseModule.set_hyperparameters
.. automethod:: parametricmatrixmodels.modules.BaseModule.get_num_trainable_floats
.. automethod:: parametricmatrixmodels.modules.BaseModule.get_state
.. automethod:: parametricmatrixmodels.modules.BaseModule.set_state
.. automethod:: parametricmatrixmodels.modules.BaseModule.set_precision
.. automethod:: parametricmatrixmodels.modules.BaseModule.astype
.. automethod:: parametricmatrixmodels.modules.BaseModule.serialize
.. automethod:: parametricmatrixmodels.modules.BaseModule.deserialize

