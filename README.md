# George Replication

This repository contains a Jupyter notebook that implements the George method for training robust deep neural networks. The George method aims to mitigate the problem of deep neural networks exploiting non-predictive features that are spuriously correlated with class labels. The method follows a 3-step pipeline:

1. Train a model using Empirical Risk Minimization (ERM).
2. Cluster inputs based on the output they produce for ERM.
3. Retrain using "Group-Balancing" to ensure in each batch, each group appears equally.

## Table of Contents
- [Dataset](#dataset)
- [Results](#results)
- [References](#references)

## Dataset

The notebook uses the SpuCoMNIST dataset, which is a modified version of the MNIST dataset with added spurious features. The dataset is initialized and loaded within the notebook.

## Results

The notebook provides evaluation metrics for the model trained using ERM and for the model retrained using the group-balancing method. It includes group-wise accuracy, worst group accuracy, and average accuracy.

## References

- George Method: [George: A method for mitigating spurious correlations in deep neural networks](https://arxiv.org/abs/2011.12945)
- SpuCo Quickstart Notebooks: [SpuCo Quickstart Notebooks](https://github.com/BigML-CS-UCLA/SpuCo/tree/master/quickstart/spuco_mnist)
