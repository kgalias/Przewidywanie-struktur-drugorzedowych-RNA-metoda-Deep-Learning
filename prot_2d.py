import pickle
from generators import generator
from create_model import create_model
from keras.callbacks import EarlyStopping
   
nb_aminoacid = 21
nb_structure = 8
batch_size = 32
nb_epoch = 16

# load in protein data 
with open('sequences.pickle', 'rb') as handle:
    sequences = pickle.load(handle)
# load in secondary structure data    
with open('structures.pickle', 'rb') as handle:
    structures = pickle.load(handle)

# establish length of longest sequence
max_len = max([len(seq) for seq in sequences])

# set 10% of the data to be used for validation
validation_len = len(sequences) // 10

# as the data might not fit in memory, define generators for both the training and validation data
train_generator = generator(sequences[:-validation_len], structures[:-validation_len], max_len, batch_size)
test_generator = generator(sequences[-validation_len:], structures[-validation_len:], max_len, batch_size)

# stop training if validatation loss goes up
early_stopping = EarlyStopping(monitor='val_loss', patience=2)


# create the first model
model = create_model(max_len, nb_aminoacid, nb_structure, 1, 64, 10, 0.)
model.summary()

# train the model
hist = model.fit_generator(train_generator, 
                           samples_per_epoch=128, 
                           nb_epoch=nb_epoch, 
                           callbacks=[early_stopping], 
                           validation_data=test_generator, 
                           nb_val_samples=32)

# save accuracy and loss data for each epoch
with open('hist_conv1d.pickle', 'wb') as handle:
    pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# create the second model
model = create_model(max_len, nb_aminoacid, nb_structure, 2, 64, 10, 0.)
model.summary()

# train the model
hist = model.fit_generator(train_generator, 
                           samples_per_epoch=128, 
                           nb_epoch=nb_epoch, 
                           callbacks=[early_stopping], 
                           validation_data=test_generator, 
                           nb_val_samples=32)

# save accuracy and loss data for each epoch
with open('hist_conv1d_2x.pickle', 'wb') as handle:
    pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)    

# create the third model
model = create_model(max_len, nb_aminoacid, nb_structure, 1, 64, 10, 0.2)
model.summary()

# train the model
hist = model.fit_generator(train_generator, 
                           samples_per_epoch=128, 
                           nb_epoch=nb_epoch, 
                           callbacks=[early_stopping], 
                           validation_data=test_generator, 
                           nb_val_samples=32)

# save accuracy and loss data for each epoch
with open('hist_conv1d_dropout.pickle', 'wb') as handle:
    pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

# create the fourth model
model = create_model(max_len, nb_aminoacid, nb_structure, 2, 64, 10, 0.2)
model.summary()

# train the model
hist = model.fit_generator(train_generator, 
                           samples_per_epoch=128, 
                           nb_epoch=nb_epoch, 
                           callbacks=[early_stopping], 
                           validation_data=test_generator, 
                           nb_val_samples=32)

# save accuracy and loss data for each epoch
with open('hist_conv1d_2x_dropout.pickle', 'wb') as handle:
    pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)            