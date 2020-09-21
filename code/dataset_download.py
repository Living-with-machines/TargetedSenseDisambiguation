import json,pickle
import requests
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser

def query_oed(
          auth:dict,
          endpoint:str,
          query:str,
          flags:str='',
          level:str='',
          verbose=True):
    
    """
    Get data from Oxford English Dictionary.
    Function requires an endpoint _and_ query as arguments. 
    
    Arguments:
    
        auth (dict): a dictionary with authentication inforamtion, needs details for 'app_id' and 'app_key'
    
        endpoint (str): select which endpoint to query, examples are word, sense, semanticclass etc
        
        query (str): query for the specific endpoint, most often a specific id, such as 'machine_nn01' or '120172'
        
        flags (str): options appended to query to include, for example, quotations instead of quotation ids
                     example "include_senses=false&include_quotations=false"
        
        level (str): at which level to query the endpoint, 
                     e.g. get sense of the query word, get siblings for semantic class etc
                     standard value is empty string
        
        verbose (bool): print the URL used for retrieving information from the API
        
    Returns:
    
        JSON of the response
        
    Example uses:
    
        query_oed(auth, 'word', 'machine_nn01')
            -> Retrieves information for the word machine_nn01.
        
        query_oed(auth, 'word','machine_nn01',level='quotations')
            -> Retrieves all quotations for machine_nn01.
        
        query_oed(auth, 'word', 'machine_nn01', flags="include_senses=true&include_quotations=true")
            -> Retrieves all senses and quotations  for the word machine_nn01.
            
        query_oed(auth, 'semanticclass', '163378')
            -> Retrieves semantic class with id 163378.
            
        query_oed(auth, 'semanticclass', '163378', level='children')
            -> Retrieves all children for the semanticlass with id 163378.
            
        query_oed(auth,'semanticclass', '163378', level='branchsenses',flags="current_in='1750-1950'")
            -> get all senses (siblings _and_ descendants) branching out from semantic class with id 163378
               restrict query to all senses observed between 1750 and 1950.
                
    """
    
    base_url = "https://oed-researcher-api.oxfordlanguages.com/oed/api/v0.2"
    
    url = f"{base_url}/{endpoint}/{query}" # build url
    
    if level: # if a level has been specified add this to the url
        url = f"{url}/{level}/"
    
    if flags: #  add flag to url with a question mark
        url = f"{url}?{flags}"
        
    response = requests.get(url, headers=auth) 
        
    if response.status_code == 200: # check status code 

        data = json.dumps(response.json())
        senses = json.loads(data)

        return senses
    
    else:
        raise Exception(f"Error while accessing the API\nResponse code={response.status_code}")


def convert_json_to_dataframe(senses):

    """
    Receive json of the OED API response
    Return a dataframe
    """    

    senses_overview = pd.DataFrame()

    for item in senses['data']["senses"]:
        senses_overview = senses_overview.append(pd.io.json.json_normalize(item))

    return senses_overview

def parse_input_commands():
    """
    read inputs from the command line
    return the lemma_id
    """    

    parser = ArgumentParser()
    parser.add_argument("-l", "--lemmaid", help="The lemma id to be used for creating the dataframe",)
    args = parser.parse_args()
    lemma_id = args.lemmaid
    if lemma_id:
        return lemma_id
    else:
        parser.exit("ERROR: The lemma id is missing, you should query it for instance using -l machine_nn01")

# an overall dictionary of ids: descriptions to avoid asking multiple times for the same id to the API
overall_sem_class_ids_dict = {}

def semantic_class_ids_to_descriptions(semantic_class_lists,credentials):
    """
    read list of lists of semantic_class_ids and credentials for the API
    return a dictionary of ids: descriptions
    """    

    sem_class_dict = {}

    # this is divided in a list of lists - do we want to keep it?
    # for the moment is converted in a single dictionary for each row
    for semantic_class_ids in semantic_class_lists:
        for sem_id in semantic_class_ids:
            if sem_id in overall_sem_class_ids_dict:
                sem_class_dict[sem_id] = overall_sem_class_ids_dict[sem_id]
            else:
                url_sem = "https://oed-researcher-api.oxfordlanguages.com/oed/api/v0.2/semanticclass/" + str(sem_id)
                r_sem = requests.get(url_sem, headers=credentials)

                if r_sem.status_code == 200: # check status code         
                    entry_json_sem = json.dumps(r_sem.json())
                    entry_sem = json.loads(entry_json_sem)
                    sem_class_dict[sem_id] = entry_sem['data']['label']
                    overall_sem_class_ids_dict[sem_id] = entry_sem['data']['label']       
                else:
                    print ("Missing description for the following sense_id:", sem_id)
                    overall_sem_class_ids_dict[sem_id] = None       
                    sem_class_dict[sem_id] = None

    return sem_class_dict

lemma_id = parse_input_commands()

with open('../oed_experiments/oed_credentials.json') as f:
    credentials = json.load(f)

#query the API and get the json response
sense_json = query_oed(credentials,'word',lemma_id,'include_senses=true&include_quotations=true')

# convert the json in a dataframe
senses_df = convert_json_to_dataframe(sense_json)

# convert semantic class ids to labels
senses_df['semantic_class_ids'] = senses_df['semantic_class_ids'].apply(lambda x: semantic_class_ids_to_descriptions(x,credentials))

# save the dataframe
save_path = Path("../data")
save_path.mkdir(exist_ok=True)

senses_df.to_pickle(save_path / (lemma_id+"_senses.pickle"))  

senses_df.to_csv(save_path / (lemma_id+"_senses.tsv"),sep='\t')
