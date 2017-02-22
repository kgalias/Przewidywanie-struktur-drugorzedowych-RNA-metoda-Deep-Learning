import numpy as np
from keras import backend as K
import theano.tensor as T

def categorical_crossentropy_no_pad(y_true, y_pred):
    original_pad = y_true.shape[1]
    loss = K.zeros_like(y_true[:, :, 0])
   
    y_true = y_true.reshape((-1, y_true.shape[-1]))
    y_pred = y_pred.reshape((-1, y_pred.shape[-1]))
    
    indices = T.neq(y_true.sum(axis=1), 0)
    
    loss = indices * K.categorical_crossentropy(y_pred, y_true)
    loss = loss.reshape((-1, original_pad)).mean(axis=1)
    
    return loss    

def categorical_accuracy_no_pad(y_true, y_pred):
    assert(y_true.ndim == 3)
    assert(y_pred.ndim == 3)
    
    y_true = y_true.reshape((-1, y_true.shape[-1]))
    y_pred = y_pred.reshape((-1, y_true.shape[-1]))
    assert(y_true.ndim == 2)
    assert(y_pred.ndim == 2)
    
    zero_indices = K.equal(K.sum(y_true, axis=1), 0)
    assert zero_indices.ndim == 1, zero_indices.ndim
   
    aa = K.argmax(y_true, axis=1)
    bb = K.argmax(y_pred, axis=1)

    no_of_padding = zero_indices.sum()
    
    return (K.sum(K.equal(aa, bb)) - no_of_padding) / (y_true.shape[0] - no_of_padding)