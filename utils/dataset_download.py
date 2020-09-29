import json,pickle
import requests
import pandas as pd
from collections import defaultdict
from tqdm.notebook import tqdm
from pathlib import Path, PosixPath
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



def traverse_thesaurus(auth:dict,
                  query_df:pd.DataFrame,
                  save_to:PosixPath=Path("./data"),
                  start:int=1750,
                  end:int=1950):
    """
    TO DOs:
     - get quotations
     - merge all information into one dataframe

    Given a dataframe with senses of a specific lemma
    This function attempts to find all sibling and descendants 
    of the last semantic class (the leaf) of each sense.
    
    the start and end argument allow to define a date range
    these years are the added to current_in flag
    
    Arguments:
        auth (dict): authenticationn credentials for the OED API
        query_df (pd.DataFrame): pandas dataframe with and export of the OED API
        save_to (PosixPath): where to store the output
        start (int): sense should be current from this year
        end (int): sense should be current until this year
    
    Returns
        a dictionary which maps an semantic class idx 
        to an array of senses that are siblings and descendants
    """

    # get all leaves of paths shown in semantic_class_ids
    # the last item of the lists in the semantic_class_ids columnns
    semanticclass_ids = set([sc[-1] for scs in query_df.semantic_class_ids.to_list() for sc in scs])
    # use branchsenses of the semanticclass endpoint
    # this return an "array of senses that belong to the 
    # semantic class specified by ID, plus senses that 
    # belong to its child and descendant classes." 
    # according the OED API documentation
    responses = {idx : query_oed(auth,'semanticclass', idx, level='branchsenses',flags=f"current_in='{start}-{end}'")
                     for idx in semanticclass_ids}
    with open(save_to / 'tree_traversal.pickle','wb') as out_pickle:
        pickle.dump(responses,out_pickle)
    
    return responses

def get_quotations_from_thesaurus(auth:dict,tt:dict):
    """
    This functions gets all quotations for senses retrieved using
    the traverse_thesaurus function. Information retrieved with 
    this function can be merged later with the tree traversal
    using __enter_function_name__.
    
    Arguments:
        auth (dict): authenticationn credentials for the OED API
        tt (dict): a tree traversal dict, generated by the traverse_thesaurus function
         which uses the branchsensses level of the semanticclass endpoint for find
         map the lowest semantic class to all the sense ids
         
    Returns:
        a nested dictionary which maps semantic class ids
        for a dictionary which, in turn, maps sense ids to 
        the quoations listed in the OED.
     
    """
    # get a set of tuples with all the sense idx in the first position
    # and the semantic class they figure in in the second position
    senses_with_semantic_class = set(
                              (sense.get('id'),sc_idx,) 
                                    for sc_idx in tt.keys() 
                                        for sense in tt.get(sc_idx,{}).get('data',[])
                                       )
                                    
    
    # map all sense ids to a the quotations
    sense_idx2quotations = {
                sense_idx : query_oed(auth,'sense',sense_idx,level='quotations')
                        for sense_idx,semantic_class_idx in tqdm(senses_with_semantic_class)
                                }
    
    # create an empty dictionary which will map
    # semantic class ids to a list in which
    # each element is again a dictionary, but
    # on that maps sense idx to the actual quotations
    sem_class_idx2senses = defaultdict(list)
    

    for sense_idx,sem_class_idx in senses_with_semantic_class:
        # append {sense_id : quoations} to list under key sem_class_idx
        sem_class_idx2senses[sem_class_idx].append(
                         sense_idx2quotations[sense_idx]
                            )
    # store output    
    with open('./data/tree_traversal_quotations.pickle','wb') as out_pickle:
        pickle.dump(sem_class_idx2senses,out_pickle)
    
    return sem_class_idx2senses

def merge_pickled(seed_query, tree_traversal, tree_quotations):
    """Function that merges all information.
    Arguments:
        seed_query (PosixPath): ...
        tree_traversal (PosixPath): ...
        tree_quotations (PosixPath): ...
        
        
    Returns:
        ... 
    """
    def reshape_word_export(df):
        """Helper function to reshape information
        obtain via de word endpoint
        """
        rows = []

        for i,row in df.iterrows():    
            for quotation in row.quotations:
                quot_dict = dict(quotation)
                quot_dict.update(dict(row))
                quot_dict['sense_id'] = row.id
                quot_dict["id_quotation"] = quot_dict.pop("id")
                rows.append(quot_dict)

        return pd.DataFrame(rows)
    
    with open(seed_query,'rb') as seed_pickle:
        root = pickle.load(seed_pickle)
    
    with open(tree_traversal,'rb') as tree_pickle:
        tree = pickle.load(open(tree_traversal,'rb'))
        
    with open(tree_quotations,'rb') as quotations_pickle:
        quotations = pickle.load(quotations_pickle)
        
    tree_df = pd.concat(
                    [pd.DataFrame.from_dict(sense,orient='index').T 
                            for key in tree.keys() 
                                 for sense in tree[key]['data']
                            ]).reset_index(drop=True, inplace=False)
        
    quotations_df = pd.concat(
                        [pd.DataFrame.from_dict(sense['data'],orient='columns')
                            for key in quotations.keys() 
                                 for sense in quotations[key]
                            ]).reset_index(drop=True, inplace=False)

    merged_df = tree_df.merge(quotations_df[['sense_id','id','source','text']],
                           left_on='id', right_on='sense_id',suffixes=('','_quotation'))
    merged_df['root'] = False # distinguish root senses for extended senses
    
    root_df = reshape_word_export(root)
    root_df["root"] = True # distinguish root senses for extended senses
    
    columns = list(set(merged_df.columns).intersection(set(root_df.columns)))
    
    return pd.concat([root_df[columns],merged_df[columns]])
        

if __name__ == "__main__":

    lemma_id = parse_input_commands()

    with open('../oed_experiments/oed_credentials.json') as f:
        credentials = json.load(f)

    #query the API and get the json response
    sense_json = query_oed(credentials,'word',lemma_id,'include_senses=true&include_quotations=true')

    # convert the json in a dataframe
    senses_df = convert_json_to_dataframe(sense_json)

    # convert semantic class ids to labels
    #senses_df['semantic_class_ids'] = senses_df['semantic_class_ids'].apply(lambda x: semantic_class_ids_to_descriptions(x,credentials))

    # save the dataframe

    save_path = Path("../data")
    save_path.mkdir(exist_ok=True)

    senses_df.to_pickle(save_path / f"senses_{lemma_id}.pickle")

    senses_df.to_csv(save_path / f"senses_{lemma_id}.tsv",sep='\t')

