#                 TP 2– Outils pour l’apprentissage automatique : Réseaux de neurones avec Keras
# Question 3 : Préparation des données et construction d’un réseau de neurones
#  Nom et Prénom : Abderrahim Bouhriz
# Groupe des alternants ILSEN

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
from keras import optimizers
from keras.layers import Dropout
from time import time
from keras.callbacks import TensorBoard, ModelCheckpoint

import pandas as pnd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)




#************************ Recuperation des données ******************#

dataset = pnd.read_csv('acoustique_voy_orales_20loc_ESTER_NCCFr_contexte_freqLex_distCentroide.txt',
sep='\t', usecols=['voyelle', 'F1', 'F2', 'F3', 'F4', 'Z1', 'Z2', 'f0'])

# *********************** Netoyage de Corpus *************************#

dataset = dataset[dataset['F4'] != '--undefined--'] # supprimer les lignes avec F4 = --undefined--
dataset.dropna() #suprimer les lignes null 
dataset[['F1', 'F2', 'F3', 'F4','Z1', 'Z2', 'f0']] = dataset[['F1', 'F2', 'F3', 'F4','Z1', 'Z2', 'f0']].astype(float) # rendre toutes les données de type float

# *********************** Extraction des données *********************#

y = dataset.voyelle # Extraire les données de la colonne des voyelles (La sortie)
x = dataset.drop('voyelle', axis=1) # Extraire les colonnes 'F1', 'F2', 'F3', 'F4', 'Z1', 'Z2', 'f0' comme données d'entrainements.

# ********************** Transformation de données *******************#

x = x.values #numpy array
# Cet estimateur met à l'échelle et traduit chaque caractéristique individuellement de manière à ce qu'elle se situe dans la plage donnée sur l'ensemble d'apprentissage, par exemple entre zéro et un.
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x) # Ajustez aux données, puis transformez-les
x = pnd.DataFrame(x_scaled) # Structure de données étiquetées à deux dimensions avec des colonnes de types potentiellement différents.
le = LabelEncoder() # Coder les étiquettes cibles avec une valeur comprise entre 0 et n_classes-1.
y = le.fit_transform(y)

#*********************** Preparation des donnees **********************#
 
x_train, x_data, y_train, y_data = train_test_split(x, y,test_size=0.2) # Donnees d’apprentissage (80%)
x_test, x_validation, y_test, y_validation = train_test_split(x_data, y_data,test_size=0.5) # (20%) Donnees restantes = Donnees de validation (10%) + Donnees de test (10%)

output_size = len(pnd.unique(dataset['voyelle'])) # return la longeur des voyelles de valeurs uniques ( output_size = 10 ) 
#  transformer ces nombres en vecteurs appropriés pour les utiliser avec des modèles. Pour la classification
y_train = to_categorical(y_train, output_size)
y_test = to_categorical(y_test, output_size)
y_validation = to_categorical(y_validation, output_size) 


#****************** Initialisation des callbacks pour Tensorbard *******# 
log_dir="logs/{}".format(time())
checkpoint = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

#********************** Utiliser modelchekpoint ************************#
#Utilisez modelchekpoint pour enregistrer le meilleur score dans le répertoire tmp/checkpoint 
checkpoint_filepath = 'tmp/checkpoint' + 'meilleurversion.{epoch:02d}-{val_accuracy:.2f}.hdf5'
model_Checkpoint_Callback = ModelCheckpoint(checkpoint_filepath,
monitor='val_accuracy', verbose=1,
save_best_only=True, save_weights_only=False,
mode='max', period=1)

# ********************** Paramétrer le modèle **********************************#

model = Sequential() # initialisation du model

# Nous prendrons 7 dans la couche d'entrée car nous avons 7 valeurs :  'F1', 'F2', 'F3', 'F4','Z1', 'Z2', 'f0'

model.add(Dense(142, input_dim=7, activation='relu'))

model.add(Dense(130, activation='relu'))
# Supprimer aléatoirement et temporairement des neurones du réseau.
model.add(Dropout(0.2))


# La couche de sortie contient 10 sorties (10 classes). Et comme il s'agit d'une classification, nous allons utiliser l'activation softmax.
model.add(Dense(output_size, activation='softmax'))

# Compiler le modèle 
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=["accuracy"])

model.fit(x_train,y_train,epochs=20,verbose=1,validation_data=(x_validation, y_validation), callbacks=[checkpoint,model_Checkpoint_Callback])

#*************************** Evaluation de model *********************************#
score = model.evaluate(x_test, y_test, verbose=1)
print('Test_score:', score[0])
print('Test_accuracy:', score[1])

# Test_score: 1.1066457033157349
# Test_accuracy: 0.5962913036346436