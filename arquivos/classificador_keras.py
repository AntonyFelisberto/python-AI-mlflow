import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.utils import to_categorical
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.datasets import mnist

import mlflow
import mlflow.tensorflow

(x_treinamento,y_treinamento),(x_teste,y_teste) = mnist.load_data()
plt.imshow(x_treinamento[29],cmap="gray")
plt.title(y_treinamento[29])
plt.show()

x_treinamento = x_treinamento.reshape((len(x_treinamento),np.prod(x_treinamento.shape[1:])))
x_teste = x_teste.reshape((len(x_teste),np.prod(x_teste.shape[1:])))
print(x_teste)

x_treinamento = x_treinamento.astype("float32")
x_teste = x_teste.astype("float32")
x_treinamento /= 255
x_teste /= 255
y_treinamento = to_categorical(y_treinamento,10)
y_teste = to_categorical(y_teste,10)
print(y_teste)

def treina_dl(n_camadas_ocultas,n_units,activation,drop_out,epochs):
    try:
        mlflow.create_experiment("DLExperimento")
    except:
        pass
    mlflow.set_experiment("DLExperimento")

    with mlflow.start_run():
        mlflow.tensorflow.autolog()

        mlflow.set_tag("n_camadas_ocultas",n_camadas_ocultas)
        mlflow.set_tag("n_units",n_units)
        mlflow.set_tag("activation",activation)
        mlflow.set_tag("drop_out",drop_out)
        mlflow.set_tag("epochs",epochs)

        modelo = Sequential()

        #camadas oculta mais camada de entrada
        modelo.add(Dense(units=n_units,activation=activation,input_dim=784))
        modelo.add(Dropout(drop_out))

        #camada oculta adicional
        for n in range(n_camadas_ocultas):
            modelo.add(Dense(units=n_units,activation=activation))
            modelo.add(Dropout(drop_out))

        modelo.add(Dense(units=10,activation="softmax"))

        modelo.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

        modelo.summary()

        historico = modelo.fit(x_treinamento,y_treinamento,epochs=epochs,validation_data=(x_teste,y_teste))

        historico.history.keys()
        loss = plt.plot(historico.history["val_loss"])
        plt.savefig("loss.png")
        acuracia = plt.plot(historico.history["val_accuracy"])
        plt.savefig("acuracia.png")

        mlflow.log_artifact("loss.png")
        mlflow.log_artifact("acuracia.png")

        print("modelo: ", mlflow.active_run().info.run_uuid)
    mlflow.end_run()


n_camadas_ocultas=[1,2,3]
n_units=[16,32,64]
activation=["relu","tanh"]
drop_out=[0.1,0.2]
epochs=[5,10,20]
for camada in n_camadas_ocultas:
    for unidades in n_units:
        for ativacao in activation:
            for drop in drop_out:
                for epocas in epochs:
                    treina_dl(camada,unidades,ativacao,drop,epocas)