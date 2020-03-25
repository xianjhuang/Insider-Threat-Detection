import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D,Dropout,Conv2D
from keras.applications import MobileNetV2,vgg16
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam, RMSprop, SGD
import keras.metrics

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
#base_model=vgg16.VGG16(input_shape=(32, 32, 3),weights='imagenet',include_top=False,classes=2,pooling='avg') #imports the mobilenet model and discards the last 1000 neuron layer.

model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
#model.add(layers.Dense(512, activation='relu'))#,kernel_regularizer=regularizers.l2(0.001)))
#model.add(layers.Dropout(0.5))

#model.add(layers.Dropout(0.5))
model.add(layers.Dense(1024, activation='relu'))#,kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1024, activation='relu'))#,kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))
#
for layer in base_model.layers:
        layer.trainable=False
#        
#x=base_model.output
#print("Shape of X ...........   ",x.shape)
#
#x=GlobalAveragePooling2D()(x)
#print("Shape of GlobalAveragePooling2D ...........   ",x.shape)
#x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
#x = Dropout(0.5)(x)
#
#x=Dense(512,activation='relu')(x) #dense layer 2
##x = Dropout(0.5)(x)
#
##x=Dense(512,activation='relu')(x) #dense layer 3
#x = Dropout(0.5)(x)
##
#preds=Dense(2,activation='softmax')(x) #final layer with softmax activation
##
####specify the inputs
####specify the outputs
#model=Model(inputs=base_model.input,outputs=preds)
#
##now a model has been created based on our architecture
print(model.summary())
#

#for i,layer in enumerate(base_model.layers):
#  print(i,layer.name)

#layers = [(layer, layer.name, layer.trainable) for layer in model.layers] pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])

#print(layers)

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_mobilenet.png', show_shapes=True, show_layer_names=True)

# In[5]:
TRAIN_DIR = 'C:\\Users\\gradhabaigopina\\Downloads\\\CERT_Data\\CERT_Data_Ratio_5'
PRED_DIR = 'C:\\Users\\gradhabaigopina\\Downloads\\CERT_Data\\Test_Data'

data_gen=ImageDataGenerator(validation_split=0.3,preprocessing_function=preprocess_input) #included in our dependencies
#
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

#
#test_datagen = ImageDataGenerator()
#test_generator = test_datagen.flow_from_directory(PRED_DIR,
#                                                 target_size=(32,32),
#                                                 color_mode='rgb',
#                                                 batch_size=32,
#                                                 class_mode='categorical',
#                                                 shuffle=False)
import collections
train_Count = train_generator.classes

print("Training.......Number of records.......",collections.Counter(train_Count))

Val_Count = validation_generator.classes

print("Validation....Number of records.........",collections.Counter(Val_Count))

opt = SGD(lr=0.000001, momentum=0.99)
#opt=RMSprop(lr=0.000001)
model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])#,keras.metrics.Precision(),keras.metrics.Recall()]) #,tf.keras.metrics.AUC(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy
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
