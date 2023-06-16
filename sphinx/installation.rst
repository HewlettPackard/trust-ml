============
Installation
============

To install, clone `our code from GitHub <https://github.com/HewlettPackard/trust-ml/>`_
and then install its dependencies into your (virtual) environment from :code:`requirements.txt` and
:code:`requirements-tensorrt.txt` as follows:

.. code-block:: bash
    
    pip install -r requirements.txt
    pip install --extra-index-url=https://pypi.ngc.nvidia.com --trusted-host pypi.ngc.nvidia.com -r requirements-tensorrt.txt

Note that Python 3.8 or newer is required.
