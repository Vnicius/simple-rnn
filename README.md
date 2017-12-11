# simples-rnn
A simple Recurrent Neural Network with TensorFlow.

The code get the dataset mnist of TensorFlow to train a Neural Network with dimensions defined by parameters.

*Based in [this](https://www.youtube.com/watch?v=dFARw8Pm0Gk) tutorial video*
*A modification of [this](https://github.com/Vnicius/simple-multilayer-nn) code*

## Dependences

- Python = 3.x
- TensorFlow = 1.4

## Run

```console
    $ python3 nn.py [1] [2] [3]
```

1. Number of recurrences
2. Number of epochs of train
3. The size of batch to train

## Examples

**Running:**
```console
    $ python3 nn.py 128 3 128
```

**Expected output**
```console
    Extracting /tmp/data/train-images-idx3-ubyte.gz
    Extracting /tmp/data/train-labels-idx1-ubyte.gz
    Extracting /tmp/data/t10k-images-idx3-ubyte.gz
    Extracting /tmp/data/t10k-labels-idx1-ubyte.gz
    Epoch: 1 of 3
    Loss: 193.482811514

    Epoch: 2 of 3
    Loss: 56.1591507513

    Epoch: 3 of 3
    Loss: 38.9748192206

    Accuracy: 97.6700007915%
```
