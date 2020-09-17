import json,pickle
import requests
import pandas as pd

def query_oed(endpoint:str,
          query:str,
          flags:str='',
          level:str='',
          verbose=True):
    """Get data from Oxford English Dictionary
    Arguments:
        endpoint (str): select which endpoint to query, examples are word, sense, semanticclass etc
        query (str): query for the specific endpoint, most often a specific id, such as 'machine_nn01' or '120172'
        flags (str): options appended to query to include, for example, quotations instead of quotation ids
                     example "include_senses=false&include_quotations=false"
        level (str): at which level to query the endpoint, 
                     e.g. get sense of the query word, get siblings for semantic class etc
                     standard value is empty string
    Returns:
        json of the response
    """
    
    base_url = "https://oed-researcher-api.oxfordlanguages.com/oed/api/v0.2"
    url = f"{base_url}/{endpoint}/{query}" # build url
    
    if flags and level:
        raise Exception("Define either flag or level\nThese options can not be used in combination")
    
    if level: # if a level has been specified add this to the url
        url = f"{url}/{level}/"
    
    if flags:
        url = f"{url}?{flags}"

    response = requests.get(url, headers = {"app_id": app_id, "app_key": app_key}) 
    
    if verbose:
        print(url)
        
    if response.status_code == 200: # check status code 

        data = json.dumps(response.json())
        senses = json.loads(data)

        senses_overview = pd.DataFrame()
        for item in senses['data']["senses"]:
            senses_overview = senses_overview.append(pd.io.json.json_normalize(item))

        return senses_overview
    
    else:
        raise Exception(f"Error while accessing the API\nResponse code={response.status_code}")

with open('../oed_experiments/oed_credentials.json') as f:
    credentials = json.load(f)

app_id = credentials["app_id"]
app_key = credentials["app_key"]

lemma_id = "machine_nn01"

senses_df = query_oed('word',lemma_id,'include_senses=true&include_quotations=true')

senses_df.to_pickle(lemma_id+"_senses_oed.pickle")  

senses_df.to_csv(lemma_id+"_senses_oed.tsv",sep='\t')
