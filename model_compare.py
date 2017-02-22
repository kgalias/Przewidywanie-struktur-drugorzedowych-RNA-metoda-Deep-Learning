import pickle
import matplotlib.pyplot as plt

with open('hist_oh_conv1d.pickle', 'rb') as handle:
    hist_oh_cond1d = pickle.load(handle)

# summarize history for accuracy
plt.plot(hist_oh_cond1d['categorical_accuracy_no_pad'])
plt.plot(hist_oh_cond1d['val_categorical_accuracy_no_pad'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(hist_oh_cond1d['loss'])
plt.plot(hist_oh_cond1d['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()