#                 TP 2– Outils pour l’apprentissage automatique : Réseaux de neurones avec Keras
# Question 2 partie 2: programme qui charge ce réseau et traite le corpus de test sans réaliser un nouvel apprentissage.
#  Nom et Prénom : Abderrahim Bouhriz
# Groupe des alternants ILSEN

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from keras import optimizers
from time import time
from keras.callbacks import TensorBoard, ModelCheckpoint

# la meilleure version du model
model_best_version = 'tmp/checkpointmeilleurversion.20-0.98.hdf5'
model = Sequential()
# charger la meilleure version du model
model = load_model(model_best_version)
# téléchargement de données
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Normalisation des donnée et avoir des valeurs entre 0 et 1
x_train, x_test = x_train / 255.0, x_test / 255.0


x_test = x_test.reshape(10000, 784)


#Modifier les valeurs entières des étiquettes en vecteurs de type 1-hot
y_test = to_categorical(y_test, 10)

#Evaluation du model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test_score:', score[0])
print('Test_accuracy:', score[1])



# Résultats:
#Test_score: 0.11766314506530762
#Test_accuracy: 0.9819999933242798

