import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn

#DIGITE NO CONSOLE PARA ENTRAR NA UI DO MLFLOW: mlflow ui
credito = pd.read_csv("arquivos\\Credit.csv")
print(credito.shape)
print(credito.head())

for col in credito.columns:
    if credito[col].dtype == "object":
        credito[col] = credito[col].astype("category").cat.codes

previsores = credito.iloc[:,0:20].values
classe = credito.iloc[:,20].values

x_treinamento,x_teste,y_treinamento,y_teste = train_test_split(previsores,classe,test_size=0.3,random_state=123)

try:
    mlflow.create_experiment("rfexperimento")
except:
    pass

mlflow.set_experiment("rfexperimento")

def treina_rf(n_estimators):
    with mlflow.start_run():
        modelorf = RandomForestClassifier(n_estimators=n_estimators)
        modelorf.fit(x_treinamento,y_treinamento)
        previsoes = modelorf.predict(x_teste)

        mlflow.log_param("n_estimators",n_estimators)

        # Metricas
        acuracia = accuracy_score(y_teste, previsoes)
        recall = recall_score(y_teste, previsoes)
        precisao = precision_score(y_teste, previsoes)
        f1 = f1_score(y_teste, previsoes)
        auc = roc_auc_score(y_teste, previsoes)
        log = log_loss(y_teste, previsoes)

        mlflow.log_metric("acuracia", acuracia)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precisao", precisao)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("log", log)

        confusion = confusion_matrix(y_teste, previsoes)
        plt.figure()
        plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xticks([0, 1], [0, 1])
        plt.yticks([0, 1], [0, 1])
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.savefig("confusionrf.png")

        roc = roc_curve(y_teste, previsoes)
        plt.figure()
        plt.plot(roc[0], roc[1])
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.savefig("rocrf.png")

        mlflow.log_artifact("confusionrf.png")
        mlflow.log_artifact("rocrf.png")

        mlflow.sklearn.log_model(modelorf, "ModeloNB")

        print("modelo: ", mlflow.active_run().info.run_uuid)

    mlflow.end_run()

arvores = [50,100,500,750,100]
for n in arvores:
    treina_rf(n)