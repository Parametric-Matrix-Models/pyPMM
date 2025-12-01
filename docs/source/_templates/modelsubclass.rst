{{ name | escape | underline }}

``{{ fullname }}``

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   :members:
   :special-members: __call__, compile, grad_input, grad_params
   :exclude-members: __init__, _abc_impl

   .. automethod:: __init__

   .. autosummary::
      :nosignatures:

      __call__
      _get_callable
      {% for item in methods %}
      {% if not item.endswith('__init__') %}
         ~{{ name }}.{{ item }}
      {%- endif %}
      {%- endfor %}

   .. autosummary::

      {% for item in attributes %}
      {% if not item.endswith('__init__') %}
         ~{{ name }}.{{ item }}
      {%- endif %}
      {%- endfor %}
