HyperPower (Keras/TensorFlow + Spearmint)
===========================
<img align="right" src="http://users.ece.cmu.edu/~dstamoul/image/Picture4.png" width="100px"/>

Hardware-aware hyper-parameter search for [Keras](https://keras.io/)+[TensorFlow](https://www.tensorflow.org/) neural networks
via [Spearmint](https://github.com/HIPS/Spearmint) Bayesian optimisation.


Description
-----------

Hyper-parameter optimization of neural networks (NN) has emerged as a challenging process.
This design problem becomes more daunting if we are find the optimal (in terms of classification error)
NN configuration that also satisfies hardware constraints, e.g., maximum inference runtime,
maximum GPU energy or power consumption.

HyperPower uses the effectiveness of Bayesian optimization to employ
hardware-constrained hyper-parameter optimization. This codebase is as basis
in the [HyperPower paper](https://arxiv.org/abs/1712.02446):

```
HyperPower: Power- and Memory-Constrained Hyper-Parameter Optimization for Neural Networks
Dimitrios Stamoulis, Ermao Cai, Da-Cheng Juan, Diana Marculescu
Design, Automation and Test in Europe Conference & Exhibition (DATE), 2018. IEEE, 2018.
```

General Setup
-------------

**STEP 1: Installation (prerequisites)**

1. Install [Keras](https://keras.io/) and [TensorFlow](https://www.tensorflow.org/)
2. Make sure you have `nvidia-smi` (to read GPU power/energy values)
3. Download and install [MongoDB](https://www.mongodb.org/)
4. Install [Spearmint](https://github.com/HIPS/Spearmint).

**STEP 2: Experimental setup**

1. Defining the neural network (and the hyper-parameters) in Keras

    An example of a neural network can be found in [`experiments/cifar10/keras_model`](experiments/cifar10/keras_model/network_def.py).
    In a default Keras model (w/o hyper-parameter optimization) the syntax would be:
    ```
    <...>
    momentum_val = 0.9
    decay_val = 0.001
    lr_val = 0.01

    opt=keras.optimizers.SGD(lr=lr_val, decay=decay_val, momentum=momentum_val)
    <...>
    ```

    To define hyper-parameters, we instead write:

    ```
    <...>
    momentum_val = HYPERPARAM{"type":"FLOAT", "token": "momentum", "min": 0.95, "max": 0.999}
    decay_val = HYPERPARAM{"type":"INT", "token": "weight_decay_base", "transform": "NEGEXP10", "min": 1, "max": 5}
    lr_val = HYPERPARAM{"type":"INT", "token": "base_lr_base", "transform": "NEGEXP10", "min": 1, "max": 5}

    opt=keras.optimizers.SGD(lr=lr_val, decay=decay_val, momentum=momentum_val)
    <...>
    ```

    For example, the learning rate is defined as a hyper-parameter `HYPERPARAM{"type":"INT", "token": "base_lr_base", "transform": "NEGEXP10", "min": 1, "max": 5}`,
    with range from 1 to 5; by using a transformation key `NEGEXP10`, this corresponds to taking the negative of the exponent of these values (1, ..., 5) with base 10,
    i.e., Spearmint will try {0.1, ..., 0.0001}. For more information on the transformations, we utilize a syntax similar to the one used in [CWSM](https://github.com/kuz/caffe-with-spearmint).
    Please note that our syntax also requires the `token` entry to be defined, i.e., a unique name for each hyper-parameter.


2. Generating Spearmint-compatible config file

    The provided [`gener_experiment.py`](gener_experiment.py) file parses the `experiments/myexperiment/keras_model/network_def.py`
    file and generates the config.json file directory with the definitions of the hyper-parameters exposed to Spearmint.

    You should make sure that you set the `SPEARMINT_ROOT` variable in the `gener_experiment.py` script so that
    `SPEARMINT_ROOT/spearmint/main.py` can be launched from the `gener_experiment.py` script.


3. Callable objective function.

    The provided [`hyperpower.py`](hyperpower.py) gets the hyper-parameters of the design instance suggested next by Spearmint
    and translates this to a callable Keras script. The script is executed to obtain (i) the overall classification error
    and (ii) the inference runtime/power/energy per image of the NN.


**STEP 2: Run hardware-aware Bayesian optimization**

You are ready to optimize! We provide a hyper-parameter optimization example
in [`experiments/cifar10/`](experiments/cifar10/). You can launch your hardware-constrained
hyper-parameter optimization with the provided [`run_tool.sh`](run_tool.sh).


