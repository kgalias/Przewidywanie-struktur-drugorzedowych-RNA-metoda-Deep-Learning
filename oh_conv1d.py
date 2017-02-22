import pickle
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Convolution1D, InputLayer, TimeDistributed, Dense, Masking, Lambda
from no_pad import categorical_crossentropy_no_pad, categorical_accuracy_no_pad


res_to_num = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'E': 5, 'Q': 6, 'G': 7, 'H': 8, 'I': 9,  
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'X': 20}

struct_to_num = {' ': 0, 'H': 1, 'B': 2, 'E': 3, 'G': 4, 'I': 5, 'T': 6, 'S': 7}    
    
no_of_aas = len(res_to_num)
no_of_labels = len(struct_to_num)
no_of_samples = 100 # for testing purposes, whole doesn't fit in memory anyways, use fit_generator() (?)
    
with open('sequences.pickle', 'rb') as handle:
    sequences = pickle.load(handle)
    
with open('structures.pickle', 'rb') as handle:
    structures = pickle.load(handle)

sequences_one_hot = np.array([np_utils.to_categorical([res_to_num[s] for s in seq], no_of_aas) for seq in sequences[:no_of_samples]])
structures_one_hot = np.array([np_utils.to_categorical([struct_to_num[s] for s in struct], no_of_labels) for struct in structures[:no_of_samples]])

X = pad_sequences(sequences_one_hot)
y = pad_sequences(structures_one_hot)

print(np.shape(X))
print(np.shape(y))

model = Sequential()
model.add(InputLayer(input_shape=(415, no_of_aas))) # 415 = max length in first 100 samples
model.add(Convolution1D(64, 10, border_mode='same'))
model.add(TimeDistributed(Dense(no_of_labels, activation='softmax')))
model.compile(loss=categorical_crossentropy_no_pad, optimizer='Adam', metrics = [categorical_accuracy_no_pad])
model.summary()
hist = model.fit(X, y, batch_size=64, nb_epoch=10, validation_split=0.1)

with open('hist_oh_conv1d.pickle', 'wb') as handle:
    pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
'''
l = 0
min_l, max_l = 10000, 0
for r in sequences:
    if 'Z' in r:
        print(r)
    l_r = len(r)
    l += l_r
    if l_r < min_l:
        min_l = l_r
    if l_r > max_l:
        max_l = l_r
print("length of all: ", l, "\n", end=" ")
print("min length: ", min_l, "\n", end=" ") 
print("max length: ", max_l, "\n", end=" ") 
print("avg length: ", l / len(sequences), "\n", end=" ") 
'''
