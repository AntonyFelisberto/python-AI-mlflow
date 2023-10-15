import requests
import pandas as pd

credito = pd.read_csv("arquivos\\Credit.csv")

for col in credito.columns:
    if credito[col].dtype == "object":
        credito[col] = credito[col].astype("category").cat.codes

dados = credito.iloc[0:10,0:20].to_json(orient="split")

previsao = requests.post(url="http://localhost:2345/invocations",headers={"Content-Type": "application/json",},data=dados)
print(previsao.text)