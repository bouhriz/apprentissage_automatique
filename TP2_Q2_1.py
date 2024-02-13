#                 TP 2– Outils pour l’apprentissage automatique : Réseaux de neurones avec Keras
# Question 2 partie 1: mécanisme de sauvegarde d’un modèle lorsque celui-ci améliore la métrique choisie sur le corpus d’évaluation
#  Nom et Prénom : Abderrahim Bouhriz
# Groupe des alternants ILSEN

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
from keras import optimizers
from time import time
from keras.callbacks import TensorBoard, ModelCheckpoint


#Initialisation des callbacks pour Tensorbard 
log_dir="logs/{}".format(time())
checkpoint = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)


#Utiliser modelchekpoint pour enregistrer le meilleur score dans le répertoire tmp/checkpoint : 
checkpoint_filepath = 'tmp/checkpoint' + 'meilleurversion.{epoch:02d}-{val_accuracy:.2f}.hdf5'
model_Checkpoint_Callback = ModelCheckpoint(checkpoint_filepath,
monitor='val_accuracy', verbose=1,
save_best_only=True, save_weights_only=False,
mode='max', period=1)

model = Sequential() # initialisation du model

#création de la 1er couche cachée de 200 noeud  avec 784 entrées car on a des dim de 28*28 , la fonction d'activation est 'relu'
model.add(Dense(200, input_dim=784, activation='relu'))

#Création de la 2ème couche cachée de 200 noeud  avec 200 entré , la fonction d'activation est 'relu'
model.add(Dense(200, activation='relu'))


#Création de la couche sortie (output)  avec 200 entré et de 10 noeud , la fonction d'activation est 'softmax'
model.add(Dense(10, activation='softmax'))

# Compiler le modèle 
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=["accuracy"])


#telechargement des données
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalisation des donnée et avoir des valeurs entre 0 et 1
x_train, x_test = x_train / 255.0, x_test / 255.0


x_train = x_train.reshape(60000, 784) # cette ligne va revoir les corpus train utilisé pour l'apprentissage du model avec  vecteur du taille 784
x_test = x_test.reshape(10000, 784)  # la même chose que l'ancienne mais cette fois avec test.

# modification des valeurs entières des étiquettes pour avoir les valeurs dans un vecteur one-hot, pour résoudre ce problème de classification
y_train = to_categorical(y_train, 10) 
y_test = to_categorical(y_test, 10)



model.fit(x_train,y_train,epochs=20,verbose=1,validation_data=(x_test, y_test), callbacks=[checkpoint,model_Checkpoint_Callback])

# Evaluation 
score = model.evaluate(x_test, y_test, verbose=0)
print('Test_score:', score[0])
print('Test_accuracy:', score[1])


# Adam est le meilleur optimiseur 
# Résultats:
# Test_score: 0.11003411561250687
# Test_accuracy: 0.9810000061988831

