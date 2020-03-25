
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D,Dropout,Conv2D
from keras.applications import MobileNetV2,vgg16
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam, RMSprop,SGD,Adamax
import keras.metrics

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import balanced_accuracy_score
from keras import models , regularizers, layers, optimizers, losses, metrics
print(keras.__version__)

# Fixed for our Cats & Dogs classes
NUM_CLASSES = 2

# Fixed for Cats & Dogs color images
CHANNELS = 3

RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'
#



# create the base pre-trained model
base_model = keras.applications.ResNet50(input_shape=(32,32,3),weights='imagenet',include_top=False)

model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
#model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dropout(0.5))
#model.add(layers.Dense(1024, activation='relu'))#,kernel_regularizer=regularizers.l2(0.01)))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(1024, activation='relu'))#,kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))#),kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(2, activation='softmax'))

#for layer in base_model.layers[:]:
 #  layer.trainable = False


for layer in base_model.layers[:165]:
   layer.trainable = False
for layer in base_model.layers[165:]:
   layer.trainable = True

print('Last block of the conv_base is now trainable')

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_resnet.png', show_shapes=True, show_layer_names=True)


TRAIN_DIR = 'C:\\Users\\gradhabaigopina\\Downloads\\\CERT_Data\\CERT_Data_Ratio_100'
PRED_DIR = 'C:\\Users\\gradhabaigopina\\Downloads\\CERT_Data\\Test_Data'

data_gen=ImageDataGenerator(validation_split=0.2,preprocessing_function=preprocess_input) #included in our dependencies
#
train_generator=data_gen.flow_from_directory(TRAIN_DIR,
                                                 target_size=(32,32),
                                                 color_mode='rgb',
                                                 batch_size=128,
                                                 class_mode='categorical',
                                                 shuffle=True, 
                                                 subset='training')
#
validation_generator = data_gen.flow_from_directory(TRAIN_DIR,
                                                 target_size=(32,32),
                                                 color_mode='rgb',
                                                 batch_size=128,
                                                 class_mode='categorical',
                                                 shuffle=False,
                                                 subset='validation')

import collections
train_Count = train_generator.classes

print("Training.......Number of records.......",collections.Counter(train_Count))

Val_Count = validation_generator.classes

print("Validation....Number of records.........",collections.Counter(Val_Count))
#opt = Adamax(lr=0.000001)#, momentum=0.9)
opt = RMSprop(lr=0.00001)
model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])#,keras.metrics.Precision(),keras.metrics.Recall()]) #,tf.keras.metrics.AUC(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
print(model.summary())

# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy
step_size_train=train_generator.n//train_generator.batch_size

#Use the fit_generator method of the ImageDataGenerator class to train the network.
epochs = 5

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

#Y_pred = model.predict_generator(test_generator, (test_generator.n) // train_generator.batch_size+1)
# Get classes by np.round
#Confution Matrix and Classification Report
Y_pred = model.predict_generator(validation_generator, validation_generator.n // train_generator.batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = ['Malicious', 'Non-Malicious']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

val_trues = validation_generator.classes

#print("val_preds .........  ", val_preds)
#print("val_trues .......... ",val_trues)

## accuracy: (tp + tn) / (p + n)
#accuracy = accuracy_score(val_trues, val_preds)
#print('Accuracy: %f' % accuracy)
## precision tp / (tp + fp)
precision = precision_score(val_trues, y_pred)
print('Precision: %f' % precision)
## recall: tp / (tp + fn)
recall = recall_score(val_trues, y_pred)
print('Recall: %f' % recall)
## f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(val_trues, y_pred)
print('F1 score: %f' % f1)
#
## kappa
kappa = cohen_kappa_score(val_trues, y_pred)
print('Cohens kappa: %f' % kappa)
## ROC AUC
##auc = roc_auc_score(val_trues, Y_pred)
##print('ROC AUC: %f' % auc)
## confusion matrix
#matrix = confusion_matrix(val_trues, val_preds)
#print(matrix)
#
tn, fp, fn, tp = confusion_matrix(val_trues, y_pred).ravel()
print("True negative ....   ",tn)
print("True positive ....   ",tp)

print("False positive ....   ",fp)
print("False negative ....   ",fn)
#
#print(classification_report(val_trues, val_preds))
#
#
bal_acc = balanced_accuracy_score(val_trues, y_pred)
print('Balanced Accuracy: %f' % bal_acc)
#
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(val_trues, y_pred)

plt.plot(recall, precision, label=thresholds)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall')
plt.legend(loc="upper right")
plt.savefig("prec-recall.png")
plt.show()
#
#
