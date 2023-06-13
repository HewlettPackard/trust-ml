=====
Usage
=====



Dataset Structure
-----------------

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


Run
---
```

python main.py --model_path ./models/imagenet/architecture/resnet50_imagenet1000.pk --weight_path ./models/imagenet/weights/resnet50_imagenet1000.pt --dataset imagenet --dataset_path ./datasets/imagenet/val --log_dir ./results/imagenet/resnet50

```


