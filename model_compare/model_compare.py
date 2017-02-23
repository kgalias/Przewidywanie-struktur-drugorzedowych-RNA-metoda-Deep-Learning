import pickle
import matplotlib.pyplot as plt

with open('hist_conv1d.pickle', 'rb') as handle:
    hist_conv1d = pickle.load(handle)
with open('hist_conv1d_2x.pickle', 'rb') as handle:
    hist_conv1d_2x = pickle.load(handle)
with open('hist_conv1d_dropout.pickle', 'rb') as handle:
    hist_conv1d_dropout = pickle.load(handle)
#with open('hist_conv1d_2x_dropout.pickle', 'rb') as handle:
#    hist_conv1d_2x_dropout = pickle.load(handle)
    
# summarize history for accuracy for model one
plt.plot(hist_conv1d['categorical_accuracy_no_pad'])
plt.plot(hist_conv1d['val_categorical_accuracy_no_pad'])
plt.title('model accuracy for 1 conv layer')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss for model one
plt.plot(hist_conv1d['loss'])
plt.plot(hist_conv1d['val_loss'])
plt.title('model loss for 1 conv layer')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for accuracy for model two
plt.plot(hist_conv1d_2x['categorical_accuracy_no_pad'])
plt.plot(hist_conv1d_2x['val_categorical_accuracy_no_pad'])
plt.title('model accuracy for 2 conv layers')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss for model two
plt.plot(hist_conv1d_2x['loss'])
plt.plot(hist_conv1d_2x['val_loss'])
plt.title('model loss for 2 conv layers')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for accuracy for model three
plt.plot(hist_conv1d_dropout['categorical_accuracy_no_pad'])
plt.plot(hist_conv1d_dropout['val_categorical_accuracy_no_pad'])
plt.title('model accuracy for 1 conv layer w/ dropout')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss for model three
plt.plot(hist_conv1d_dropout['loss'])
plt.plot(hist_conv1d_dropout['val_loss'])
plt.title('model loss for 1 conv layer w/ dropout')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for accuracy for model three
plt.plot(hist_conv1d_dropout['categorical_accuracy_no_pad'])
plt.plot(hist_conv1d_dropout['val_categorical_accuracy_no_pad'])
plt.title('model accuracy for 1 conv layer w/ dropout')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss 
plt.plot(hist_conv1d['val_loss'])
plt.plot(hist_conv1d_2x['val_loss'])
plt.plot(hist_conv1d_dropout['val_loss'])
#plt.plot(hist_conv1d_dropout['val_loss'])
plt.title('valuation loss comparison')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['conv', '2 conv', 'conv w/ dropout'], loc='upper left')
plt.show()

# summarize history for accuracy for model three
plt.plot(hist_conv1d['val_categorical_accuracy_no_pad'])
plt.plot(hist_conv1d_2x['val_categorical_accuracy_no_pad'])
plt.plot(hist_conv1d_dropout['val_categorical_accuracy_no_pad'])
plt.title('valuation accuracy comparison')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['conv', '2 conv', 'conv w/ dropout'], loc='upper left')
plt.show()
