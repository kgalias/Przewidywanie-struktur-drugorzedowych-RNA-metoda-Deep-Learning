from keras.models import Sequential
from keras.layers import Convolution1D, InputLayer, TimeDistributed, Dense, Dropout
from no_pad import categorical_crossentropy_no_pad, categorical_accuracy_no_pad

def create_model(max_len, nb_aminoacid, nb_structure, nb_layer=1, nb_filter=64, filter_length=10, p=0.2):
    model = Sequential()
    model.add(InputLayer(input_shape=(max_len, nb_aminoacid)))
    for i in range(nb_layer):
        model.add(Convolution1D(nb_filter, filter_length, border_mode='same'))
        model.add(Dropout(p))
    model.add(TimeDistributed(Dense(nb_structure, activation='softmax')))
    model.compile(loss=categorical_crossentropy_no_pad, optimizer='Adam', metrics=[categorical_accuracy_no_pad])
    return model