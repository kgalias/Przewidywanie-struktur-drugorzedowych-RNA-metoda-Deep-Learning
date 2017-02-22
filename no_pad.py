#import numpy as np
from keras import backend as K
from keras.backend.common import _EPSILON
import theano.tensor as T

def categorical_crossentropy_no_pad(y_true, y_pred):
    original_pad = y_true.shape[1]
    loss = K.zeros_like(y_true[:, :, 0])
   
    y_pred = K.clip(y_pred, _EPSILON, 1-_EPSILON)
   
    y_true = y_true.reshape((-1, y_true.shape[-1]))
    y_pred = y_pred.reshape((-1, y_pred.shape[-1]))
    
    indices = T.neq(y_true.sum(axis=1), 0)
    
    loss = indices * K.categorical_crossentropy(y_pred, y_true)
    loss = loss.reshape((-1, original_pad)).mean(axis=1)
    
    return loss    

def categorical_accuracy_no_pad(y_true, y_pred):
    n = y_true.shape[-1]
    y_true = y_true.reshape((-1, n))
    y_pred = y_pred.reshape((-1, n))
    
    nonzero_indices = K.not_equal(K.sum(y_true, axis=1), 0)
   
    aa = K.argmax(y_true, axis=1)
    bb = K.argmax(y_pred, axis=1)

    return K.sum(K.equal(aa, bb) * nonzero_indices)  / (nonzero_indices.sum())