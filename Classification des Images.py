#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importation des librairies necessaires
from keras.datasets import cifar10
import matplotlib.pyplot as plt


# In[2]:


#Importer et diviser le dataset cifar10
(train_X,train_Y),(test_X,test_Y)=cifar10.load_data()


# In[4]:


#Tracer certaines images de l’ensemble de données pour visualiser l’ensemble de données
n=6
plt.figure(figsize=(20,10))
for i in range(n):
    plt.subplot(330+1+i)
    plt.imshow(train_X[i])
    plt.show()


# In[5]:


#Importer les librairies pour creer notre architecture CNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils


# In[6]:


#Convertir les valeurs de pixels pour l’ensemble de données en type float, puis normaliser l’ensemble de données
train_x=train_X.astype('float32')
test_X=test_X.astype('float32')
 
train_X=train_X/255.0
test_X=test_X/255.0


# In[7]:


#Mettre les donnees en One-hot-Encoding
train_Y=np_utils.to_categorical(train_Y)
test_Y=np_utils.to_categorical(test_Y)
 
num_classes=test_Y.shape[1]


# In[8]:


#Créer le modèle séquentiel et ajouter les couches
model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(32,32,3),
    padding='same',activation='relu',
    kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32,(3,3),activation='relu',padding='same',kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512,activation='relu',kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


# In[11]:


#Configurer l’optimiseur et compiler le modèle
sgd=SGD(lr=0.01,momentum=0.9,decay=(0.01/25),nesterov=False)
 
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])


# In[12]:


#Le résumé du modèle pour une meilleure compréhension de l’architecture du modèle
model.summary()


# In[13]:


#Entrainer le modele
model.fit(train_X,train_Y,validation_data=(test_X,test_Y),epochs=10,batch_size=32)


# In[14]:


#Calculer l'accuracy pour le testset 
_,acc=model.evaluate(test_X,test_Y)
print(acc*100)


# In[16]:


#Enregister notre modele 
model.save("modele_classification_images.h5")


