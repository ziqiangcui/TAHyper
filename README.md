# TAHyper-SDM24
Code for the SDM'2024 paper, [Treatment-Aware Hyperbolic Representation Learning for Causal Effect Estimation with Social Networks](https://arxiv.org/abs/2401.06557).

The code is based on [network-deconfounder-wsdm20]https://github.com/rguo12/network-deconfounder-wsdm20 and [hgcn](https://github.com/HazyResearch/hgcn).


### Dependencies

```
Python 3.8
Pytorch 2.0.0+cu118
Scipy 1.10.0
Numpy 1.24.4
Pandas 2.0.3
```

### Datasets

Datasets used in this paper can be found in ```./datasets```

### Running the experiment

On a linux system, you can run the main.py to train and evaluate the model, for example

```
python main.py
```

