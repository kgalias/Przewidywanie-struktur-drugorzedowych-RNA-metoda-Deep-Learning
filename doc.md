###Description
Uses keras to train some models for protein secondary structure prediction (including a neural network with one convolutional layer, two convolutional layers, possibly with dropout).
###Prerequisites
Python (e.g. from [Anaconda](https://www.continuum.io/downloads))  
Keras and dependencies (e.g. using ```conda install -c anaconda keras=1.1.1```)  
Theano (see [Installing Theano](http://deeplearning.net/software/theano/install.html))
###Usage
```python prot_2d.py``` trains the nets and saves the history. ```python model_compare.py``` produces the graphs.
###Comparisons for some simple models
![](https://github.com/kgalias/protein-2d-prediction/blob/master/model_compare/conv1d_acc.png)
![](https://github.com/kgalias/protein-2d-prediction/blob/master/model_compare/conv1d_loss.png)

![](https://github.com/kgalias/protein-2d-prediction/blob/master/model_compare/conv1d_dropout_acc.png)
![](https://github.com/kgalias/protein-2d-prediction/blob/master/model_compare/conv1d_dropout_loss.png)

![](https://github.com/kgalias/protein-2d-prediction/blob/master/model_compare/conv1d_2x_acc.png)
![](https://github.com/kgalias/protein-2d-prediction/blob/master/model_compare/conv1d_2x_loss.png)

![](https://github.com/kgalias/protein-2d-prediction/blob/master/model_compare/loss_comp.png)
![](https://github.com/kgalias/protein-2d-prediction/blob/master/model_compare/acc_comp.png)
