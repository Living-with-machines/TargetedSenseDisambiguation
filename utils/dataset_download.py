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
        verbose=False):
    
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
    
    if verbose:
        print(url)
        
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

        
def get_provenance_by_semantic_class(row: pd.Series) -> list:
    """
    decide on the relation between the sense and the target querry
    here we use the lowest semantic class id to decide on the relation
    
    if last semantic class id (sc_ids[-1]) == provenance id: then sense is sibling of provenance id
    elif provenance semantic class id in the list of semantic class last ids
    (but provenance not the last one): then sense is descendant of provenance id
    Argument:
        row (pd.Series): row of dataframe obtained from branchsenses endpoint
    
    Returns:
        nested listed in the format of [lowest semantic class id, relation, provenance semantic class id]
            in other words it said that for a given sense (which can have multiple semantic class ids)
            the lowest semantic class id stands in the relation "sibling" or "descendant" of the 
            provenance semantic class id
    """
    
    provenance = []
    
    # one sense can belong to multiple semantic class ids
    for sc_ids in row.semantic_class_ids:
        relation = ''
        
        # scenario 1
        # if the last id equals provenance, the relation is sibling
        if sc_ids[-1] == row.provenance_pivot:
            relation = 'sibling'
        
        # scenario 2
        # if not, then the relation is descendant
        elif (row.provenance_pivot in sc_ids):
            relation = 'descendant'
        
        # exclude other relations
        if relation:
            provenance.append([sc_ids[-1], relation, row.provenance_pivot])
    
    # double check, each sense SHOULD have a provenance
    # if not this will print a warning message
    if not provenance:
        print(f'Warning: No descendants or siblings found for {row.id}')

    return provenance

def extend_from_lemma(auth: dict, 
                    lemma_id: str,
                    start:int=1750,
                    end:int=1950) -> pd.DataFrame:
    
    
    """Extends senses from a dataframe created from information obtained
    via the OED API word endpoint. The script first retrieves all
    senses (either from saved pickle or from the API),
    then obtains synonyms for these senses, then continues to dowload 
    other senses that based on their the semantic classes.
    
    This script also aims to record the "provenance" of words, 
    their relation to the initial query, which can help to 
    select of filter words. 
    
    Filtering words based on their provenance is handled by the
    filter_senses_by_provenance function.
    
    Arguments:
        auth (dict): a dictionary with authentication inforamtion, needs details for 'app_id' and 'app_key'
        
        lemma_id (str): define lemma id, assumes this refers to a saved pickle file in the format ./data/sense_{lemma_id}.pickle
        
        start (int): define start year for harvesting senses used for filtering senses using the 'current_in' flag
        
        end (int): define end year for harvesting senses used for filtering senses using the 'current_in' flag
    Returns
    
        a pandas.DataFrame with extended number of senses
    """    
    
    # helper function that, given a nested list, creates a new list with the last element of each list in the nested list
    get_last_id = lambda nested_list :[l[-1] for l in nested_list]
    
    # load seed query dataframe or download from api
    lemma_path = f"./data/senses_{lemma_id}.pickle"
    
    if Path(lemma_path).is_file():
        print(f'Loading senses for {lemma_id} from pickle.')
        query_df = pd.read_pickle(lemma_path)
    else:
        print(f'Dowloading senses for {lemma_id} from OED API.')
        sense_json = query_oed(auth,'word',lemma_id,flags='include_senses=true&include_quotations=true')
        # convert the json in a dataframe
        query_df = convert_json_to_dataframe(sense_json)
        # save the datafram as pickle
        query_df.to_pickle(f"./data/senses_{lemma_id}.pickle")
    
    # from here on we _only_ use the sense endpoint to ensure all information 
    # can be properly concatenated in one dataframe
    
    # retrieve all sense ids
    query_sense_ids = query_df.id.unique()
    
    # get all senses by sense id
    print(f"Get all sense for the lemma {lemma_id}")
    seeds = [(s,query_oed(auth,'sense',s,
                    flags=f"current_in='{start}-{end}'&limit=1000", # probably "current_in" not needed here see APi
                    verbose=False)) # set verbose to True to see the url request
                        for s in tqdm(query_sense_ids)]
    
    # convert to dataframe
    seeds_df = pd.DataFrame([seed['data'] for s_id,seed in seeds])
    
    # seed_df contains all the senses of the word machine_nn01
    # we distinguish between provenance and provenance_type
    # provenance will refer to specific word, sense of semantic class ids
    # provenance_type will distinguish between different types of extension
    # define provenance, these words are "seed"
    seeds_df['provenance'] = [[[i,'seed',lemma_id]] for i in seeds_df.id] # for the seed sense we use the id of the word machine_nn0
                                    # we use list here, reason is explained later, see provenance of synonyms
    seeds_df['provenance_type'] = 'seed' # categorize these lemmas as seed
    
    # get all synonyms for the seed senses
    # reminder synonyms uses same function as the /senses/ endpoint, flags should work here
    print(f"Get all synonyms of the senses listed in {lemma_id}")
    synonyms = [(s,query_oed(auth,'sense',s,
                level='synonyms',
                flags=f"current_in='{start}-{end}'&limit=1000"))
                        for s in tqdm(query_sense_ids)]
    
    # transform list of synonyms to a dataframe
    synonyms_df = pd.DataFrame([s for s_id,syn in synonyms for s in syn['data']])
    
    # for synonyms the provenance_type is set to "synonym"
    synonyms_df['provenance_type'] = 'synonym'
    # for synonyms we refer the sense_id via which this synonym was retrieved
    synonyms_df['provenance'] = [[[s['id'],'synonym',s_id]] for s_id,syn in synonyms for s in syn['data']]
    
    # seed + synonyms constitute the nucleas of our query
    # these are saved in the core_df
    # shape should be 485 (synonyms senses) + 26 (seed senses)
    core_df = pd.concat([seeds_df,synonyms_df],sort=True)
    
    # branch out from there
    # we save the lowest level of the semantic_class_last_id columns
    core_df['semantic_class_last_id'] = core_df['semantic_class_ids'].apply(get_last_id)

    # retrieve all the _lowest_ (or last) semantic class ids for the core senses so far
    semantic_class_ids = set([s for l in core_df.semantic_class_last_id.to_list() for s in l])
    
    # now, we use the descendants endpoint
    # for each lowest semantic class id
    # we get all "descendants" which according the API documentation
    # returns an array of senses that belong to the semantic class
    # specified by ID, plus senses that belong to its child and descendant classes.
    print("Get all branches for seed senses and synonyms")
    branches = [(idx,query_oed(auth,'semanticclass', idx, 
                        level='branchsenses', # 
                        flags=f"current_in='{start}-{end}'&limit=1000"))
                            for idx in tqdm(semantic_class_ids)]
    
    # convert API response to dataframe
    branches_df = pd.DataFrame([s for idx,branch in branches for s in branch['data']])
    
    # ISSUE: again we have duplicate 
    # senses here, as some appear multiple time as
    # in the same semantic class (or as descendant)
    
    # provenance_type is branch with semantic class id 
    # that was use for retrieving the sense is the provenance
    branches_df['provenance_type'] = 'branch'
    
    # we create a provenance_pivot columsn, which shows
    # the semantic class id via which the sense was retrieved
    branches_df['provenance_pivot'] = [idx for idx, branch in branches for s in branch['data']]
    
    # now there are two scenarios to specify for the pro
    # both scenarios can apply to one sense
    # if last semantic class id (sc_ids[-1]) == provenance id: then sense is sibling of provenance id
    # elif provenance semantic class id in the list of semantic class last ids
    # (but provenance not the last one): then sense is descendant of provenance id
    
    branches_df['provenance'] = branches_df.apply(get_provenance_by_semantic_class,axis=1)
    
    # drop the provenance_pivot column
    branches_df.drop('provenance_pivot',axis=1,inplace=True)
    
    # concatenate core and branch senses
    # ISSUE: have a closer look at the warning message
    extended_df = pd.concat([core_df,branches_df],sort=True)

    # to check if rows match
    #extended_df.shape[0] == core_df.shape[0] + branches_df.shape[0]
    # save dataframe as pickle
    extended_df.to_pickle(f"./data/extended_{lemma_id}.pickle") 
    
    return extended_df

def harvest_quotations(auth: dict,lemma_id: str, level: str) -> pd.DataFrame:
    """
    Given a dataframe obtained via the OED sense endpoints
    retrieve all quotations for the included words or senses and save them
    as a dataframe, path ./data/quotations_{lemma_id}.pickle
    Argument:
        lemma_id (str): lemma of the seed query
        level (str): endpoint for harvesting quotatios (sense or word)
                    when using the sense endpoint we only get quotations relevant to
                    the initial lemma
    Returns:
        saves and returns a pd.DataFrame with quotations
    """
    df = pd.read_pickle(f'./data/extended_{lemma_id}.pickle')
    if level == 'sense':
        ids = set(df.id)
        suffix = 'related'
    elif level == 'word':
        ids = set(df.word_id)
        suffix = 'all'
    else:
        raise Exception("Choose 'word' or 'sense' as values for the 'level' argument")

    responses = [query_oed(auth,level, idx, level='quotations') for idx in tqdm(ids)]
    quotation_df = pd.DataFrame([q for r in responses for q in r['data']])
    quotation_df.to_pickle(f'./data/quotations_{suffix}_{lemma_id}.pickle')
    return quotation_df

def filter_by_year_range(dr: dict, target_start: int, target_end: int) -> bool:
    """
    Helper function that expects a datarange dictionary from the OED
    Is used for filter senses that are outside the historical scope 
    of the research. The date range is defined by the target_start and target_end
    arguments. If the date range of the sense has NO overlap with the
    target period, then return False, otherwise return True
    
    Arguments:
        dr (dict): daterange dict of OED
        target_start (int): start year of target period
        target_end (int): end year of target period
    
    Returns:
        return a boolean, True if there is overlap between
        the target period and the date range of the sense
    """
    # if there is not start date, set to 0
    if dr.get('start',None) is None:
        sense_start = 0
    else:
        sense_start = dr['start']
    
    
    # if there is no end date, set to 2021
    if dr.get('end',None) is None:
        sense_end = 2021
    else:
        sense_end = dr['end']
    
    # if there is an intersection between the target period and sense period empty
    # return True
    if set(range(sense_start,sense_end+1)).intersection(set(range(target_start,target_end+1))):
        return True
    
    # otherwise return False
    return False

def select_senses_by_provenance(sub_df: pd.DataFrame, 
                                item_ids: set, 
                                relations: list) -> tuple:
    """Helper function that given a subsection of a dataframe filters senses based
    on a set of target sense ids and relations. This function requires a dataframe created
    by the extend_from_lemma function.
    
    Arguments:
        sub_df (pd.DataFrame): slice of a pd.DataFrame
        item_ids (set): include senses related to these items 
                        these can be sense ids or semantic class ids
        relations (list): filter based on these relations 
                        options are: seed, synonyms, sibling, descedant
        
    Returns:
        a tuple that contains a list with position indices and a list with items
    """
    
    indices, items = set(),set()
    
    for i, row in sub_df.iterrows():
        for oed_id, relation, prov_id in row.provenance:
            # if the provenance and relation match to the arguments
            # add the items and position to the respective lists
            if (prov_id in item_ids) and (relation in relations):
                indices.add(i) ; items.add(oed_id)
                
    return list(indices), list(items)

def filter_senses(df, sense_ids:set, 
                    relations:list, 
                    start:int, 
                    end:int,
                    verbose=True) -> set:
    """
    Main function that filter sense by a give date range 
    and set of seed senses with provenace relations. 
    The seeds sense are selected from the lemma dataframe
    used as starting point for harvesting. Builds on dataframe created 
    by the extend_from_lemma function.
    
    Returns selected senses as a set. 
    
    Arguments:
        df (pd.DataFrame): main dataframe created by the extend_from_lemma
        senses_ids (set): seeds senses from the lemma used for filtering
        relations (list): filter based on these relations
        start (int): beginning of target period
        end (int): end of target period
        verbose (bool): print outcomes of intermediate steps
    
    Returns:
        set with senses
    """
    print("# senses before filtering by date =", df.shape[0])
    df = df[df.daterange.apply(filter_by_year_range, target_start=start, target_end=end)]
    print("# senses after filtering by date =", df.shape[0])
    
    
    seeds = df[df['provenance_type'] == "seed"].reset_index(inplace=False)
    # select words retrieved as synonyms
    # exclude those that already appear in the seed dataframe
    # reset index after selection
    synonyms = df[(df['provenance_type'] == "synonym") & (~df.id.isin(seeds.id))
                        ].reset_index(inplace=False)
    
    # select words retrieved as a branch of the synonym or a seed sense
    # exclude those that already appear as seed or synonym
    branches = df[(df['provenance_type'] == "branch") & (~df.id.isin(set(seeds.id).union(set(synonyms.id))))
                        ].reset_index(inplace=False)
    
    print("\n\n# of seed senses", seeds.shape[0],
        "\n# of synonyms", synonyms.shape[0],
        "\n# of branch senses", branches.shape[0])

    if "seed" in relations:
        seeds_selected = set(seeds[seeds.id.isin(sense_ids)].id)

    if "synonym" in relations:
        syn_sel_indices, synonyms_selected = select_senses_by_provenance(synonyms,sense_ids,relations)
    
    # as branches are retrieved by semantic class id, we get the semantic class ids 
    # of the seed AND synonyms senses
    select_seed_semantic_class_id = seeds[seeds.id.isin(seeds_selected)].semantic_class_last_id
    select_seed_semantic_class_id = set().union(*map(set,select_seed_semantic_class_id))
    
    select_synonyms_semantic_class_id = synonyms[synonyms.id.isin(synonyms_selected)].semantic_class_last_id
    select_synonyms_semantic_class_id = set().union(*map(set,select_synonyms_semantic_class_id))
    
    selected_semantic_class_id = set(select_seed_semantic_class_id).union(set(select_synonyms_semantic_class_id))
    
    branch_sel_indices, branches_selected = select_senses_by_provenance(branches,selected_semantic_class_id,relations)
    
    
    senses = set(branches.iloc[branch_sel_indices].id # for the branches we return the sense ids not the semantic class ids
            ).union(set(synonyms.iloc[syn_sel_indices].id)
                        ).union(set(seeds_selected))
    if verbose:
        print('\n\n# of seeds selected', len(seeds_selected),
            '\n# of synonyms selected', len(syn_sel_indices),
            '\n# of branches selected', len(branches_selected))
    return senses

def obtain_quotations_for_senses(
                    df_quotations:  pd.DataFrame,
                    senses: set,
                    start:int,
                    end: int
                    ) -> pd.DataFrame:
    """Create a dataframe with quotations and their metadata for 
    a selected set of senses. This function builds on
    harvest_quotations_by_sense_id.
    
    Arguments:
        df_quotations: dataframe with quotations, created using harvest_quotations_by_sense_id
        senses (set): set of senses for which we want to obtain quotations
        
    Returns:
        pd.DataFrame with quotations
        
    """
    df = pd.concat([
        pd.DataFrame.from_records(df_quotations.text.values),
        pd.DataFrame.from_records(df_quotations.source.values)
            ], axis=1)
    df['year'] = df_quotations['year']
    df['sense_id'] = df_quotations['sense_id']
    df = df[df.sense_id.isin(senses)]
    df = df[(start <= df.year) & (df.year <= end)]
    
    df.drop_duplicates(inplace=True)
    
    return df





# # an overall dictionary of ids: descriptions to avoid asking multiple times for the same id to the API
# overall_sem_class_ids_dict = {}

# def semantic_class_ids_to_descriptions(semantic_class_lists,credentials):
#     """
#     read list of lists of semantic_class_ids and credentials for the API
#     return a dictionary of ids: descriptions
#     """    

#     sem_class_dict = {}

#     # this is divided in a list of lists - do we want to keep it?
#     # for the moment is converted in a single dictionary for each row
#     for semantic_class_ids in semantic_class_lists:
#         for sem_id in semantic_class_ids:
#             if sem_id in overall_sem_class_ids_dict:
#                 sem_class_dict[sem_id] = overall_sem_class_ids_dict[sem_id]
#             else:
#                 url_sem = "https://oed-researcher-api.oxfordlanguages.com/oed/api/v0.2/semanticclass/" + str(sem_id)
#                 r_sem = requests.get(url_sem, headers=credentials)

#                 if r_sem.status_code == 200: # check status code         
#                     entry_json_sem = json.dumps(r_sem.json())
#                     entry_sem = json.loads(entry_json_sem)
#                     sem_class_dict[sem_id] = entry_sem['data']['label']
#                     overall_sem_class_ids_dict[sem_id] = entry_sem['data']['label']       
#                 else:
#                     print ("Missing description for the following sense_id:", sem_id)
#                     overall_sem_class_ids_dict[sem_id] = None       
#                     sem_class_dict[sem_id] = None

#     return sem_class_dict



# def get_branchsenses(auth:dict,
#                   lemma_id:str,
#                   query_df:pd.DataFrame,
#                   save_to:PosixPath=Path("./data"),
#                   start:int=1750,
#                   end:int=1950):
#     """
#     Given a dataframe with senses of a specific lemma
#     This function attempts to find all sibling and descendants 
#     of the last semantic class (the leaf) of each sense.
    
#     the start and end argument allow to define a date range
#     these years are the added to current_in flag
    
#     Arguments:
#         auth (dict): authenticationn credentials for the OED API
#         lemma_id (str): id of the lemma used for generating the seed dataframe
#         query_df (pd.DataFrame): pandas dataframe with and export of the OED API
#         save_to (PosixPath): where to store the output
#         start (int): sense should be current from this year
#         end (int): sense should be current until this year
    
#     Returns
#         a dictionary which maps an semantic class idx 
#         to an array of senses that are siblings and descendants
#     """

#     # get all leaves of paths shown in semantic_class_ids
#     # the last item of the lists in the semantic_class_ids columnns
#     semanticclass_ids = set([sc[-1] for scs in query_df.semantic_class_ids.to_list() for sc in scs])
#     # use branchsenses of the semanticclass endpoint
#     # this return an "array of senses that belong to the 
#     # semantic class specified by ID, plus senses that 
#     # belong to its child and descendant classes." 
#     # according the OED API documentation
#     responses = {idx : query_oed(auth,'semanticclass', idx, level='branchsenses',flags=f"current_in='{start}-{end}'&limit=1000")
#                      for idx in semanticclass_ids}
#     with open(save_to / f'branch_senses_{lemma_id}.pickle','wb') as out_pickle:
#         pickle.dump(responses,out_pickle)
    
#     return responses

# def get_synset(auth:dict,
#             lemma_id:str,
#             query_df:pd.DataFrame,
#             save_to:PosixPath=Path("./data"),
#             start:int=1750,
#             end:int=1950):
#     """
#     Given a dataframe with senses of a specific lemma
#     This function attempts to 
#         (a) get synonyms of each sense 
#         (b) get descendants of each sense + synonym senses

#     the start and end argument allow to define a date range
#     these years are the added to current_in flag
    
#     Arguments:
#         auth (dict): authenticationn credentials for the OED API
#         lemma_id (str): id of the lemma used for generating the seed dataframe
#         query_df (pd.DataFrame): pandas dataframe with and export of the OED API
#         save_to (PosixPath): where to store the output
#         start (int): sense should be current from this year
#         end (int): sense should be current until this year
    
#     Returns
#         a dictionary which maps an semantic class idx 
#         to an array of senses that are siblings and descendants
#     """

#     # get all semantic class ids for the sense in seed dataframe
#     query_semanticclass_ids = set([sc[-1] for scs in query_df.semantic_class_ids.to_list() for sc in scs])

#     # get all synonyms for each sense
#     synonyms = [query_oed(credentials,'sense',s,
#                           level='synonyms',
#                           flags=f"current_in='{start}-{end}'&limit=1000") 
#                                 for s in tqdm(query_df.id.unique())]

#     # transform list of synonyms to a dataframe
#     synonyms_df = pd.DataFrame([s for syn in synonyms for s in syn['data']])

#     # get semantic class ids of all synonym senses
#     # the last item of the list in the semantic_class_ids columnns
#     synonyms_semanticclass_ids = set([sc[-1] for scs in synonyms_df.semantic_class_ids.to_list() for sc in scs])

#     # merge semantic class ids of the query senses with synonym senses
#     semanticclass_ids = synonyms_semanticclass_ids.union(query_semanticclass_ids)

#     # get all the descendants of senses
#     descendants = [query_oed(auth,'semanticclass', idx, 
#                                 level='descendants',
#                                 flags=f"current_in='{start}-{end}'&limit=1000")
#                     for idx in semanticclass_ids]

#     descendants_df = pd.DataFrame([s for des in descendants for s in des['data']])
#     # store information
#     #with open(save_to / f'branch_synonym_senses_{lemma_id}.pickle','wb') as out_pickle:
#     #    pickle.dump(responses,out_pickle)
#     merged = pd.concat([])


#     synonyms_df.to_pickle(save_to / f'synonyms_{lemma_id}.pickle')
    
#     return responses

# def get_quotations_from_thesaurus(auth:dict,tt:dict):
#     """
#     This functions gets all quotations for senses retrieved using
#     the traverse_thesaurus function. Information retrieved with 
#     this function can be merged later with the tree traversal
#     using __enter_function_name__.
    
#     Arguments:
#         auth (dict): authenticationn credentials for the OED API
#         tt (dict): a tree traversal dict, generated by the traverse_thesaurus function
#          which uses the branchsensses level of the semanticclass endpoint for find
#          map the lowest semantic class to all the sense ids
         
#     Returns:
#         a dictionary which maps semantic class ids
#         to the quoations listed in the OED.
     
#     """
#     # get a set of tuples with all the sense idx in the first position
#     # and the semantic class they figure in in the second position
#     senses_with_semantic_class = set(
#                               (sense.get('id'),sc_idx,) 
#                                     for sc_idx in tt.keys() 
#                                         for sense in tt.get(sc_idx,{}).get('data',[])
#                                        )
                                    
    
#     # map all sense ids to a the quotations
#     sense_idx2quotations = {
#                 sense_idx : query_oed(auth,'sense',sense_idx,level='quotations')
#                         for sense_idx,semantic_class_idx in tqdm(senses_with_semantic_class)
#                                 }
    
#     # create an empty dictionary which will map
#     # semantic class ids to a list in which
#     # each element is again a dictionary, but
#     # on that maps sense idx to the actual quotations
#     sem_class_idx2senses = defaultdict(list)
    

#     for sense_idx,sem_class_idx in senses_with_semantic_class:
#         # append {sense_id : quoations} to list under key sem_class_idx
#         sem_class_idx2senses[sem_class_idx].append(
#                          sense_idx2quotations[sense_idx]
#                             )
#     # store output    
#     with open('./data/tree_traversal_quotations.pickle','wb') as out_pickle:
#         pickle.dump(sem_class_idx2senses,out_pickle)
    
#     return sem_class_idx2senses

# def merge_pickled(seed_query, tree_traversal, tree_quotations):
#     """Function that merges all information from previously
#     generated pickle files. This includes:
#         (a) Seed senses: senses retrieve via the word endpoint. 
#             This is the seed query from which we expand
#         (b) Expanded senses: expanded set of senses: using branchsenses we retrieve
#             siblings and descendants for each of the seed senses
#         (c) Quotations for expanded senses: a pickle file 
#             with all quotes for senses retrieved in (b)
#     Arguments:
#         seed_query (PosixPath): Seed senses pickle file
#         tree_traversal (PosixPath): Expanded senses pickle file
#         tree_quotations (PosixPath): Quotations for expanded senses pickle file
        
        
#     Returns:
#         a pd.DataFrame, one quotations per row 
#     """
#     def reshape_word_export(df):
#         """Helper function to reshape information
#         obtain via de word endpoint.
#         Argument:
#             df (pd.DataFrame): a dataframe derived via de work endpoint with quotation flag set to True
#         Returns:
#             a pd.DataFrame with one quotation per row
#         """
#         rows = []

#         for i,row in df.iterrows():    
#             for quotation in row.quotations:
#                 quot_dict = dict(quotation)
#                 quot_dict.update(dict(row))
#                 quot_dict['sense_id'] = row.id
#                 quot_dict["id_quotation"] = quot_dict.pop("id")
#                 rows.append(quot_dict)

#         return pd.DataFrame(rows)
    
#     with open(seed_query,'rb') as seed_pickle:
#         root = pickle.load(seed_pickle)
    
#     with open(tree_traversal,'rb') as tree_pickle:
#         tree = pickle.load(open(tree_pickle,'rb'))
        
#     with open(tree_quotations,'rb') as quotations_pickle:
#         quotations = pickle.load(quotations_pickle)
        
#     tree_df = pd.concat(
#                     [pd.DataFrame.from_dict(sense,orient='index').T 
#                             for key in tree.keys() 
#                                  for sense in tree[key]['data']
#                             ]).reset_index(drop=True, inplace=False)
        
#     quotations_df = pd.concat(
#                         [pd.DataFrame.from_dict(sense['data'],orient='columns')
#                             for key in quotations.keys() 
#                                  for sense in quotations[key]
#                             ]).reset_index(drop=True, inplace=False)

#     merged_df = tree_df.merge(quotations_df[['sense_id','id','source','text']],
#                            left_on='id', right_on='sense_id',suffixes=('','_quotation'))
#     merged_df['root'] = False # distinguish root senses for extended senses
    
#     root_df = reshape_word_export(root)
#     root_df["root"] = True # distinguish root senses for extended senses
   
#     # use only columns that appear in both root_df and merged_df
#     columns = list(set(merged_df.columns).intersection(set(root_df.columns)))
    
#     return pd.concat([root_df[columns],merged_df[columns]])
        

if __name__ == "__main__":

    lemma_id = parse_input_commands()
    
    start = 1760
    end = 1920

    with open('../oed_experiments/oed_credentials.json') as f:
        auth = json.load(f)

        
    save_path = Path("./data")
    save_path.mkdir(exist_ok=True)
    
    #query the API and get the json response
    sense_json = query_oed(auth,'word',lemma_id,flags='include_senses=true&include_quotations=true')

    # convert the json in a dataframe
    senses_df = convert_json_to_dataframe(sense_json)

    # save the dataframe
    # as pickle
    senses_df.to_pickle(save_path / f"senses_{lemma_id}.pickle")
    # as csv
    senses_df.to_csv(save_path / f"senses_{lemma_id}.tsv",sep='\t')
    
    extend_from_saved_lemma_query(auth,lemma_id,start,end)
    # get all senses that are siblings and descendants
    # of the semantic class of senses listed in previously obtained query 
    # responses = traverse_thesaurus(credentials,senses_df)
    
    # get all quoations for the senses in the responses variable
    #quotations = get_quotations_from_thesaurus(credentials,responses)
    
    # merge and save all information stored in the seperate pickle files
    # df = merge_pickled(Path(f"./data/senses_{lemma_id}.pickle"),
    #                Path("./data/tree_traversal.pickle"),
    #                Path("./data/tree_traversal_quotations.pickle"))
    
    #df.to_pickle(f"./data/{lemma_id}_all.pickle")
   