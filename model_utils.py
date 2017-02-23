from keras import backend as K
from keras.backend.common import _EPSILON
import theano.tensor as T
from keras.models import Sequential
from keras.layers import Convolution1D, InputLayer, TimeDistributed, Dense, Dropout

# cross entropy to deal with padded sequences
def categorical_crossentropy_no_pad(y_true, y_pred):
    original_pad = y_true.shape[1]
    y_pred = K.clip(y_pred, _EPSILON, 1-_EPSILON)
    loss = K.zeros_like(y_true[:, :, 0])  
   
    y_true = y_true.reshape((-1, y_true.shape[-1]))
    y_pred = y_pred.reshape((-1, y_pred.shape[-1]))
    
    nonzero_indices = T.neq(y_true.sum(axis=1), 0)
    
    loss = nonzero_indices * K.categorical_crossentropy(y_pred, y_true)
    loss = loss.reshape((-1, original_pad)).mean(axis=1)
    
    return loss    

# accuracy to deal with padded sequences   
def categorical_accuracy_no_pad(y_true, y_pred):
    n = y_true.shape[-1]
    y_true = y_true.reshape((-1, n))
    y_pred = y_pred.reshape((-1, n))
    
    nonzero_indices = K.not_equal(K.sum(y_true, axis=1), 0)
   
    aa = K.argmax(y_true, axis=1)
    bb = K.argmax(y_pred, axis=1)

    return K.sum(K.equal(aa, bb) * nonzero_indices)  / (nonzero_indices.sum())

# create model with nb_layer convolutional layers
def create_model(max_len, nb_aminoacid, nb_structure, nb_layer=1, nb_filter=64, filter_length=10, p=0.2):
    model = Sequential()
    model.add(InputLayer(input_shape=(max_len, nb_aminoacid)))
    for i in range(nb_layer):
        model.add(Convolution1D(nb_filter, filter_length, border_mode='same'))
        model.add(Dropout(p))
    model.add(TimeDistributed(Dense(nb_structure, activation='softmax')))
    model.compile(loss=categorical_crossentropy_no_pad, optimizer='Adam', metrics=[categorical_accuracy_no_pad])
    return model