Usage
=====

.. _installation:

Installation
------------

To use GeoSlide, first install it using pip:

.. code-block:: console

   (.venv) $ pip install geoslide

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``geoslide.get_random_ingredients()`` function:

.. autofunction:: geoslide.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`geoslide.get_random_ingredients`
will raise an exception.

.. autoexception:: geoslide.InvalidKindError

For example:

>>> import geoslide
>>> geoslide.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']

