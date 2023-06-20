=====
Usage
=====

Run
---
The framework can be run through :code:`main.py`. 
Example for having ResNet-50 as the victim model with which the adversarial samples can be generated:

.. code-block:: bash

    python main.py --model_path ./models/imagenet/architecture/resnet50_imagenet1000.pk \
        --weight_path ./models/imagenet/weights/resnet50_imagenet1000.pt --dataset imagenet \
        --dataset_path ./datasets/imagenet/val --log_dir ./results/imagenet/resnet50

Note that the related dataset is not included in the repository. It has to be placed manually in the dataset path that you specify.

**CLI arguments:**

* :code:`model_path`: path to a trained model
* :code:`weight_path`: path to the weights of the trained model
* :code:`dataset`: dataset name. *If missing, add the configuration for your own dataset in dataset.py (currently configured: cifar10, caltech101, imagenet)*
* :code:`dataset_path`: path to a dataset
* :code:`log_dir`: directory to log to


Generated Dataset Structure
---------------------------

The generated distorted data will have the structure following the example provided in the repository::

    └───imagenet
        └───resnet50
            └───gn
                ├───episode_c0_00003
                │       c0_x.jpg
                │       c0_x.npy
                │       c0_x_2_adv.jpg
                │       c0_x_2_adv.npy
                │       c0_x_4_adv.jpg
                │       c0_x_4_adv.npy
                │       c0_x_adv.jpg
                │       c0_x_adv.npy
                │       result_c0_00003.txt
                │
                ├───episode_c1_00011
                ...


There are both JPEG (for quick visualization) and NumPy arrays (as input to a model for benchmarking) at multiple noise
levels available with the applied distortions, as well as some metadata.


Zenodo Dataset Structure
------------------------

Datasets generated with Trust-ML are provided as pickle files. For instance,
see our data provided on `Zenodo <https://zenodo.org/record/8034833>`_.
This data should be loaded through Python's :code:`pickle` module. The
structure is a list as follows::

    Image 1, Label 1, Sample ID 1
    Image 2, Label 2, Sample ID 2
    ...
    Image N, Label N, Sample ID N

Each image is a NumPy array of shape :code:`H x W x 3`, each label is an
integer-coded class label, and each sample ID has a correspondence with a
sample in the original dataset.
