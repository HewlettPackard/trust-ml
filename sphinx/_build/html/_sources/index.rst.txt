.. trust-ml documentation master file, created by
   sphinx-quickstart on Thu Jun  1 22:48:40 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Trust-ML
========

Trust-ml is a machine learning-driven adversarial data generator that introduces naturally occurring distortions to the original dataset to generate an adversarial subset. This framework provides a custom mix of distrotions for evaluating robustness of image classification models against both true negatives and false positives.

.. figure:: images/image-1.png
   :scale: 30 %
   :alt: Difference between RLAB and competitors
   :align: center

   Difference between RLAB and competitors

RLAB unlike the existing traditional adversarial training approaches that use hand crafted attack strategy to generate adversarial samples, learns an attack strategy to generate more efficient adversarial samples. 

.. figure:: images/image-2.png
   :scale: 10 %
   :alt: RLAB Workflow
   :align: center

   RLAB Workflow

The above figure shows the overall flow of the proposed method.  Given a data sample, RLAB divides the input into a set of grid and performs sensitivity analysis.  The agent performs two actions, one to find the patch to which distortions can be added and the patch from which earlier added distortion can be removed.  This is performed iteratively until the model misclassifies the given data sample.  The final sample with the perturbations in it is called an adversarial sample that contains information about the vulnerability of the model.

.. figure:: images/Visual-Comparison.jpg
   :scale: 10 %
   :alt: RLAB's distortion comparison with Patch Attack and Square Attack from the literature
   :align: center

   RLAB's distortion comparison with Patch Attack and Square Attack from the literature



`Github Repository <https://github.com/HewlettPackard/trust-ml/>`_ of source code and documentation

.. toctree::
   :hidden:

   installation
   usage
   overview
   code
   references




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
