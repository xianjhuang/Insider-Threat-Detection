import numpy as np
import keras
import matplotlib.pyplot as plt
import keras.metrics
from keras.layers import Dense,GlobalAveragePooling2D,Dropout,Conv2D
from keras.applications import MobileNetV2,vgg16
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam, RMSprop, SGD
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import balanced_accuracy_score
from keras import models,layers,regularizers
print(keras.__version__)

image_size = 32
IMG_SHAPE = (image_size, image_size, 3)

base_model=keras.applications.MobileNetV2(input_shape=IMG_SHAPE,weights='imagenet',include_top=False,classes=2) #imports the mobilenet model and discards the last 1000 neuron layer.
model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))

for layer in base_model.layers:
        layer.trainable=False
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_mobilenet.png', show_shapes=True, show_layer_names=True)

TRAIN_DIR = 'CERT_Data_Ratio_5'
PRED_DIR = 'Test_Data'

data_gen=ImageDataGenerator(validation_split=0.3,preprocessing_function=preprocess_input) 
train_generator=data_gen.flow_from_directory(TRAIN_DIR,
                                                 target_size=(32,32),
                                                 color_mode='rgb',
                                                 batch_size=64,
                                                 class_mode='categorical',
                                                 shuffle=True, 
                                                 subset='training')
#
validation_generator = data_gen.flow_from_directory(TRAIN_DIR,
                                                 target_size=(32,32),
                                                 color_mode='rgb',
                                                 batch_size=64,
                                                 class_mode='categorical',
                                                 shuffle=False,
                                                 subset='validation')

                                              shuffle=False)
import collections
train_Count = train_generator.classes
Val_Count = validation_generator.classes
opt = SGD(lr=0.000001, momentum=0.99)
model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])

step_size_train=train_generator.n//train_generator.batch_size

#Use the fit_generator method of the ImageDataGenerator class to train the network.
epochs = 15

history = model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   validation_data=validation_generator,
                   validation_steps=50,
                   epochs=epochs)#,
#                  
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.show()

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#plt.savefig('epoch_15.png')

evalation = model.evaluate_generator(generator=validation_generator,steps=50)
print('Final test accuracy:', (evalation[1]*100.0))
validation_generator.reset()
Y_pred = model.predict_generator(validation_generator, validation_generator.n // train_generator.batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = ['Malicious', 'Non-Malicious']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

val_trues = validation_generator.classes

## precision tp / (tp + fp)
precision = precision_score(val_trues, y_pred)
print('Precision: %f' % precision)
## recall: tp / (tp + fn)
recall = recall_score(val_trues, y_pred)
print('Recall: %f' % recall)
## f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(val_trues, y_pred)
print('F1 score: %f' % f1)
