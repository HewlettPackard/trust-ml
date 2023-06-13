# Benchmark Generation Framework with Customizable Distortions for Image Classifier Robustness

This repository contains the datasets and code for the paper.

## Introduction
Trust-ML is a machine learning-driven adversarial data generator that introduces naturally occurring distortions to the
original dataset to generate an adversarial subset. This framework provides a custom mix of distortions for evaluating
robustness of image classification models against both true negatives and false positives. Our framework enables users
to audit their algorithms with customizable distortions. With the help of RLAB, we can generate more effective and
efficient adversarial samples than other distortion-based benchmarks.
In our experiments, we demonstrate that **samples generated with our framework cause greater accuracy degradation of
state-of-the-art adversarial robustness methods (as tracked by [RobustBench](https://robustbench.github.io/)) than
ImageNet-C and CIFAR-10-C**.

## Documentation and Installation
Refer to the [docs](https://hewlettpackard.github.io/trust-ml/) for documentation and installation of Trust-ML.