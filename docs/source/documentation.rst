=============
Documentation
=============

**Published at:** https://artyomzemlyak.github.io/rulka/

To contribute to the documentation, install documentation-related packages.

.. code-block:: bash

    pip install -e .[doc]

Make relevant modifications in /docs/source.

Build the updated documentation:

.. code-block:: bash

    cd docs
    make clean
    make html

Commit changes in ``docs/source/``. The ``docs/build/`` output is built by GitHub Actions and is not committed.
