# MoDe
Moment Decorrelation (MoDe) is a tool that can enforce decorrelation between some nuisance parameter (or protected attribute in ML fairness lingo) and the response of some model with gradient-based optimization (a neural network for example.) It can force trained models to have the same response across different values of the protected attribute but it can also go beyond simple decorrelation. For example, MoDe can constrain the response function to be linear or quadratic in the protected attribute. This can increase performance tremendously if the independence constraint is not necessary but instead one only cares about the "smoothness" of the dependence of the response on the protected attribute (which is a weaker constraint.)

An implementation is available in each of TensorFlow and PyTorch. The Example.ipynb notebook illustrates how MoDe is used and shows how one can obtain different response functions on a toy example.

For more details please see: https://arxiv.org/abs/2010.09745

The PyTorch implementation requires PyTorch 1.6.0 or newer. 
The Tensorflow implementation requires Tensorflow 2.2.0 or newer. 
