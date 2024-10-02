# CorrelationsRuinGD

The codebase for reproducing the simulations for the paper titled: ["Correlations are ruining your gradient descent"](https://arxiv.org/abs/2407.10780)

The codebase is setup for use with wandb for simulation tracking. To run an example of the feedback alignment algorithm, training a convolutional neural network, with decorrelation, upon the CIFAR10 dataset run:

``` 
python run.py
```

This executes a simulation as described by the parameters in `conf/config.yml`.

The virtual environment setup required to run this code is provided (for anaconda) in `environment.yml`.

## Integrating decorrelation layers into your own code

In order to use decorrelation layers within your own codebase, see the `example.ipynb` notebook for instructions.
You may also examine how they are used in the `run.py` files and how they are loaded from `crgd/decor.py`. Currently `DecorLinear` and `DecorConv2D` layers are provided.
