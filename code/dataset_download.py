import json,pickle
import requests
import pandas as pd

with open('../oed_experiments/oed_credentials.json') as f:
    credentials = json.load(f)

app_id = credentials["app_id"]
app_key = credentials["app_key"]

url = "https://oed-researcher-api.oxfordlanguages.com/oed/api/v0.2/word/machine_nn01/senses/"

senses_overview = pd.DataFrame()

r = requests.get(url, headers = {"app_id": app_id, "app_key": app_key})
data = json.dumps(r.json())
senses = json.loads(data)
for item in senses['data']:
    senses_overview = senses_overview.append(pd.io.json.json_normalize(item))

senses_overview.to_pickle("machine_senses.pickle")  
senses_overview.to_csv('machine_senses.tsv',sep='\t')
