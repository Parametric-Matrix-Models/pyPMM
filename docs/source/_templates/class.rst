{{ name | escape | underline }}

``{{ fullname }}``

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

   .. autosummary::
      :nosignatures:

      {% for item in methods %}
         ~{{ name }}.{{ item }}
      {%- endfor %}

   .. autosummary::

      {% for item in attributes %}
         ~{{ name }}.{{ item }}
      {%- endfor %}
