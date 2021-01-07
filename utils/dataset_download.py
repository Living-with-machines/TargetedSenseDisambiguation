import json,pickle
import requests
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path, PosixPath
from argparse import ArgumentParser
from typing import Union

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
    """read inputs from the command line
    """    

    parser = ArgumentParser()
    parser.add_argument("-l", "--lemma", help="The lemma to be used for creating the dataframe")
    parser.add_argument("-p", "--pos", help="The part of speech to be used for creating the dataframe")
    #parser.add_argument("-s", "--start_year", help="The start year of the data frame",default='1760')
    #parser.add_argument("-e", "--end_year", help="The end year of the data frame",default='1920')
    parser.add_argument("-d", "--download", help="use 'all' to download all quotations, 'sample' to demo the pipeline",default='sample')
    args = parser.parse_args()
    lemma = args.lemma; pos = args.pos
    #start = int(args.start_year); end = int(args.end_year)

    #if end < start:
    #    parser.exit("ERROR: 'end' should be greater than 'start'")


    penn_tags = ["CC", "CD",  "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS",
                "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR",
                "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
                "WDT", "WP", "WP$", "WRB"]
    penn_tags.extend([p.lower() for p in penn_tags])

    if pos not in penn_tags:
        parser.exit("""[ERROR] the part of speech tag {pos} is not in the
                    in the Pennn Treebank P.O.S. Tagset""")
    
    download = args.download
    if download == 'sample':
        download_all = False
    elif download == 'all':
        download_all = True
    else:
        parser.exit("[ERROR] the download argument has to be 'all' or 'sample'")
    
    if lemma:
        return lemma, pos, download_all
    else:
        parser.exit("[ERROR] The lemma is missing, you should query it for instance using -l machine_nn01")

        
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
                    lemma: str,
                    pos: str) -> pd.DataFrame:
    
    
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
    lemma_path = f"./data/lemma_senses_{lemma}_{pos}.pickle"
    
    if Path(lemma_path).is_file():
        print(f'Loading senses for {lemma}_{pos} from pickle.')
        query_df = pd.read_pickle(lemma_path)
    else:
        print(f'[LOG] Dowloading senses for {lemma}_{pos} from OED API.')
        url = f"https://oed-researcher-api.oxfordlanguages.com/oed/api/v0.2/surfaceforms/?form={lemma}&part_of_speech={pos}&limit=1000"
        response = requests.get(url, headers=auth)
        # get words with similar surface structure
        lemma_ids = set([d.get('word_id','') for d in response.json()['data']])
        print(f'[LOG] Found following word ids for {lemma}_{pos}',lemma_ids)
        # save senses for different word ids in the query_dfs list
        query_dfs = []
        for lemma_id in lemma_ids:

            sense_df = convert_json_to_dataframe(
                            query_oed(auth,'word',lemma_id,flags='include_senses=true&include_quotations=true')
                                )
                # convert the json in a dataframe
            query_dfs.append(sense_df)
        
            print(f'[LOG] Obtained {sense_df.shape[0]} senses for {lemma_id}')
    
            # save the datafram as pickle
        query_df = pd.concat(query_dfs)
        query_df.to_pickle(f"./data/lemma_senses_{lemma}_{pos}.pickle")
    
    print(f'[LOG] Obtained {query_df.shape[0]} seed senses for {lemma}_{pos}')
    # from here on we _only_ use the sense endpoint to ensure all information 
    # can be properly concatenated in one dataframe
    
    # retrieve all sense ids
    query_sense_ids = query_df.id.unique()
    
    # get all senses by sense id
    print(f"[LOG] Get all {len(query_sense_ids)} senses for the lemma {lemma}_{pos}")
    seeds = [(s,query_oed(auth,'sense',s,
                    flags=f"limit=1000", # probably "current_in" not needed here see APi
                    verbose=False)) # set verbose to True to see the url request
                        for s in query_sense_ids]
    
    # convert to dataframe
    seeds_df = pd.DataFrame([seed['data'] for _,seed in seeds])
    
    # seed_df contains all the senses of the word machine_nn01
    # we distinguish between provenance and provenance_type
    # provenance will refer to specific word, sense or semantic class ids
    # provenance_type will distinguish between different types of extension
    # define provenance, these words are "seed"
    seeds_df['provenance'] = [[[i,'seed',j]] for i,j in zip(seeds_df.id,seeds_df.word_id)] #[[[i,'seed','{lemma}_{pos}']] for i in seeds_df.id] # for the seed sense we use the id of the word machine_nn0
                                    # we use list here, reason is explained later, see provenance of synonyms
    seeds_df['provenance_type'] = 'seed' # categorize these lemmas as seed
    
    # get all synonyms for the seed senses
    # reminder synonyms uses same function as the /senses/ endpoint, flags should work here
    print(f"[LOG] Get all synonyms of the {len(query_sense_ids)} senses listed in {lemma}_{pos}")
    synonyms = [(s,query_oed(auth,'sense',s,
                level='synonyms',
                flags=f"limit=1000"))
                        for s in query_sense_ids]
    
    # transform list of synonyms to a dataframe
    synonyms_df = pd.DataFrame([s for _,syn in synonyms for s in syn['data']])
    
    # for synonyms the provenance_type is set to "synonym"
    synonyms_df['provenance_type'] = 'synonym'
    # for synonyms we refer the sense_id via which this synonym was retrieved
    synonyms_df['provenance'] = [[[s['id'],'synonym',s_id]] for s_id,syn in synonyms for s in syn['data']]
    
    # seed + synonyms constitute the nucleas of our query
    # these are saved in the core_df

    core_df = pd.concat([seeds_df,synonyms_df],sort=True)
    core_df['semantic_class_last_id'] = core_df['semantic_class_ids'].apply(get_last_id)
    print(f'[LOG] Created dateframe with {core_df.shape[0]} senses')
    core_df.to_pickle(f"./data/extended_senses_{lemma}_{pos}.pickle") 
    print(f'[LOG] Saved dataframe to "./data/extended_senses_{lemma}_{pos}.pickle"')
    # script could potentially branch out further, but we break off here for now
    return core_df
    # branch out from there
    # we save the lowest level of the semantic_class_last_id columns
    
    # retrieve all the _lowest_ (or last) semantic class ids for the core senses so far
    # semantic_class_ids = set([s for l in core_df.semantic_class_last_id.to_list() for s in l])
    
    # now, we use the descendants endpoint
    # for each lowest semantic class id
    # we get all "descendants" which according the API documentation
    # returns an array of senses that belong to the semantic class
    # specified by ID, plus senses that belong to its child and descendant classes.
    # print("Get all branches for seed senses and synonyms")
    # branches = [(idx,query_oed(auth,'semanticclass', idx, 
    #                    level='branchsenses', # 
    #                    flags=f"current_in='{start}-{end}'&limit=1000"))
    #                        for idx in tqdm(semantic_class_ids)]
    
    # convert API response to dataframe
    # branches_df = pd.DataFrame([s for idx,branch in branches for s in branch['data']])
    
    # ISSUE: again we have duplicate 
    # senses here, as some appear multiple time as
    # in the same semantic class (or as descendant)
    
    # provenance_type is branch with semantic class id 
    # that was use for retrieving the sense is the provenance
    # branches_df['provenance_type'] = 'branch'
    
    # we create a provenance_pivot columsn, which shows
    # the semantic class id via which the sense was retrieved
    # branches_df['provenance_pivot'] = [idx for idx, branch in branches for s in branch['data']]
    
    # now there are two scenarios to specify for the pro
    # both scenarios can apply to one sense
    # if last semantic class id (sc_ids[-1]) == provenance id: then sense is sibling of provenance id
    # elif provenance semantic class id in the list of semantic class last ids
    # (but provenance not the last one): then sense is descendant of provenance id
    
    # branches_df['provenance'] = branches_df.apply(get_provenance_by_semantic_class,axis=1)
    
    # drop the provenance_pivot column
    # branches_df.drop('provenance_pivot',axis=1,inplace=True)
    
    # concatenate core and branch senses
    # ISSUE: have a closer look at the warning message
    # extended_df = pd.concat([core_df,branches_df],sort=True)

    # to check if rows match
    #extended_df.shape[0] == core_df.shape[0] + branches_df.shape[0]
    # save dataframe as pickle
    #extended_df.to_pickle(f"./data/extended_{lemma_id}.pickle") 
    #print(f'Created dataframe with {extended_df.shape[0]} rows')
    #return extended_df


def harvest_data_from_extended_senses(
        auth: dict,
        lemma_pos: str,
        start_date: int=1700,
        end_date: int=2021,
        core_senses: bool = True, 
        download_all: bool = True) -> pd.DataFrame:
    """this function get definitions and quotations for all senses
    related to a surface forms that appear in the extended set of 
    senses harvested with `extend_from_lemma`. 
    
    For example if `nation_nn01-XYZ` appears in the extended set of senses, 
    this function will obtain all definitions and quotations related to the
    surface form (`nation`, NN) which includes other lemma ids
    (i.e. nation_nn01 and nation_nn01). 
    
    Note: this implies that our experiments applied to the OED are 
    optimistic estimates for out-of-sample performance,
    since we assume that lemmatized and part of speech tagging are 
    flawless when applying the the eventual models to other corpora.

    The fucntion return as dataframe with definitions and quotations
    (and some contextual information used for later filtering)

    In the process, it saves two dataframes:
        - a dataframe organized by sense
        - a dataframe organized by quotation
        the file names are prefixed with `sfrel_` (surface form related)
        to indicate that the data is retrieved via the surfaceforms endpoint
 
    Arguments:

        auth (dict): OED API creditial

        lemma_id (str): lemma_id used for harvesting extended set of senses

        start_date (int): start date for the `current_in` flag of the 
                        `surfaceforms` endpoint

        end_date (int): end date for the `current_in` flag of the 
                        `surfaceforms` endpoint

        core_senses (bool): focus only on `seed` and `synonym` senses. 
                            as extended the number of senses explodes when
                            branching out, it makes sens to only focus on
                            the core sense (senses related to the original lemma
                            and their synonyms)

        download_all (bool): use function in demo mode when False
                            for demo purpose download only the first ten lemmas
    Returns:
        a pandas.DataFrame with a unique quotation on each row
    """
    demo_suffix = ''
    df_source = pd.read_pickle(f"./data/extended_senses_{lemma_pos}.pickle") 

    if not Path(f'./data/sfrel_senses_{lemma_pos}.pickle').is_file():
        print('[LOG] Downloading data via API.')
        if core_senses:
            df_source = df_source[df_source.provenance_type.isin(['seed','synonym'])]
    
        # queries are tuples of (lemma surface form, part-of-speech)
        queries = list(set(zip(df_source.lemma,df_source.part_of_speech)))
        print(f'[LOG] Number of lemma, pos queries = {len(queries)}')
        # demo or not, demo is standard
        if not download_all:
            demo_suffix = '_demo'
            queries = queries[:10]
            print(f'[LOG] Number of lemma, pos queries = {len(queries)}')

        # container for collecting all word ids
        # related to the surface form queries
        word_ids = set()
    
        # this loop uses the OED surfaceforms endpoints
        # to retrieva all words ids associated with a surface form
        for lemma, pos in queries:
            url = f"https://oed-researcher-api.oxfordlanguages.com/oed/api/v0.2/surfaceforms/?form={lemma}&part_of_speech={pos}&current_in={start_date}-{end_date}&limit=1000"
            response = requests.get(url, headers=auth) 
            word_ids.update(set([d.get('word_id','') for d in response.json()['data']]))
    
        print(f'[LOG] Number of retrieved word ids = {len(word_ids)}')
        # for each word ids, retrieve all senses and their quotations
        data = [query_oed(auth, 'word', word_id, flags="include_senses=true&include_quotations=true") for word_id in word_ids]
        with open(f'./data/raw_output_{lemma_pos}{demo_suffix}.pickle','wb') as out_pickle:
            pickle.dump(data, out_pickle)
        
        # create a sense level dataframe
        senses_df = pd.DataFrame([s for d in data for s in d['data']['senses']])
        lemma_definition = [d['data']['definition'] for d in data for s in d['data']['senses']]
        senses_df['lemma_definition'] = lemma_definition
        senses_df.to_pickle(f'./data/sfrel_senses_{lemma_pos}{demo_suffix}.pickle')
    else:
        print('[LOG] Loading data from pickled file')
        # assume that we only want to continue from a full dataset, not a demo one
        senses_df = pd.read_pickle(f'./data/sfrel_senses_{lemma_pos}.pickle')
    
    print(f'[LOG] Shape of senses dataframe = {senses_df.shape}')

    # create a quotation level dataframe
    quotations = senses_df.explode('quotations')
    quotations.rename({'id':'sense_id'},inplace=True, axis=1)

    # make a new dataframe based on the dictionary format of the quotation column
    # the will create a dataframe in which each key of the dictionary becomes a 
    # column in the new dataframe, this dataframe will only comprise the core content
    # of each quotation 
    quotations_content = quotations.quotations.apply(pd.Series)
    quotations_content.drop({'lemma', 'oed_reference', 'oed_url', 'word_id'}, axis=1, inplace=True)
    print(f'[LOG] Shape of quotations dataframe = {quotations_content.shape}')
    # create a new dataframe with only unique definitions
    definitions = quotations[['sense_id','lemma_definition','definition','word_id','lemma']].drop_duplicates()
    final_df = definitions.merge(quotations_content[['id','source','sense_id','text','year']],on='sense_id')
    final_df.rename({'id':'quotation_id'},inplace=True, axis=1)
    print(f'[LOG] Saving pickle file to "./data/sfrel_quotations_{lemma_pos}{demo_suffix}.pickle"')
    final_df.to_pickle(f'./data/sfrel_quotations_{lemma_pos}{demo_suffix}.pickle')
    print(f'[LOG] Shape of final dataframe = {final_df.shape}')

    return final_df



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
                                level: str) -> tuple:
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
            if (prov_id in item_ids) and (relation == level):
                indices.add(i) ; items.add(oed_id)
                
    return list(indices), list(items)

def filter_senses(df, sense_ids:set, 
                    relations:Union[list,str], 
                    start:int, 
                    end:int,
                    expand_seeds:bool=True,
                    expand_synonyms:bool=True,
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
        level (str): level or depth to which to branch out, from top (seed) to bottom or branches (descendant | sibling)
                    and 'all' to return all senses within the given date range 
            options: from very specific -> all
                seed
                |_ synonym
                    |_ descendant | sibling
                        |_ all
        expand_seeds (bool): expand using semantic class ids of seed senses
        expand_synonyms (bool): expand using semantic class ids of the synonyms 
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

    if relations == 'all':
        return set(branches.id 
            ).union(set(synonyms.id)
                        ).union(set(seeds.id))
    
    print("\n\n# of seed senses", seeds.shape[0],
        "\n# of synonyms", synonyms.shape[0],
        "\n# of branch senses", branches.shape[0])


    if 'seed' in relations:
        seeds_selected = set(seeds[seeds.id.isin(sense_ids)].id)
    else:
        seeds_selected = set()


    if "synonym" in relations:
        syn_sel_indices, synonyms_selected = select_senses_by_provenance(synonyms,sense_ids,"synonym")
    else:
        syn_sel_indices, synonyms_selected = [],[]
    
    # as branches are retrieved by semantic class id, we get the semantic class ids 
    # of the seed AND synonyms senses
    branch_types = set(['sibling','descendant']).intersection(relations)
    branch_sel_indices, branches_selected = [],[]
    if branch_types: 
        select_seed_semantic_class_id = set()
        if expand_seeds:
            select_seed_semantic_class_id = seeds[seeds.id.isin(seeds_selected)].semantic_class_last_id
            select_seed_semantic_class_id = set().union(*map(set,select_seed_semantic_class_id))
        
        select_synonyms_semantic_class_id = set()
        if expand_synonyms:
            select_synonyms_semantic_class_id = synonyms[synonyms.id.isin(synonyms_selected)].semantic_class_last_id
            select_synonyms_semantic_class_id = set().union(*map(set,select_synonyms_semantic_class_id))
    
        selected_semantic_class_id = set(select_seed_semantic_class_id).union(set(select_synonyms_semantic_class_id))
        for bt in branch_types:
            bt_branch_sel_indices, bt_branches_selected = select_senses_by_provenance(branches,selected_semantic_class_id,bt)
            branch_sel_indices.extend(bt_branch_sel_indices)
            branches_selected.extend(bt_branches_selected)
    
    senses = set(branches.iloc[branch_sel_indices].id # for the branches we return the sense ids not the semantic class ids
            ).union(set(synonyms.iloc[syn_sel_indices].id)
                        ).union(set(seeds_selected))
    if verbose:
        print('\n\n# of seeds selected', len(seeds_selected),
            '\n# of synonyms selected', len(syn_sel_indices),
            '\n# of branches selected', len(branches_selected))
    return senses

def relation_to_core_senses(df):
    """
    this function creates a mapping of sense ids to their core
    i.e. synonym or seed sense id
    Arguments:
        df (pandas.DataFrame): a dataframe created by extend_from_lemma
    
    Returns:
        mapping (dict): a mapping between sense ids
    """
    scid2senseid = defaultdict(set)
    for i, row in df[df.provenance_type.isin(['seed','synonym'])].iterrows():
        for sc_id in row.semantic_class_last_id:
            # map the last sense id (via which we branched out)
            # to a sense id
            scid2senseid[sc_id].add(row.id)
            # for synonyms map this sc_id also 
            # to the provenance of the synonym
            for p in row.provenance:
                if 'synonym' in p:
                    scid2senseid[sc_id].add(p[-1])

    mapping = defaultdict(set)

    for i,row in df.iterrows(): 
        for p in row.provenance:
    
            if 'seed' in p:
                # map seed sense id to seed sense id
                # provenance is in the format of
                # [seed sense id, relation, lemma id] 
                mapping[row.id].add(p[0])
            elif 'synonym' in p:
                # map synonym id to seed sense id
                # provenance is in the format of
                # [synonym sense id, relation, seed sense id]
                # print(p)
                mapping[row.id].add(p[-1])
            elif ('sibling' in p) or ('descendant' in p):
                # map via semantic class id
                # provenance is in the shape of
                # [sem class sense, relation, sem class provenance]
                # here we map the sense id to a set 
                # of senses via scid2senseid
                # sense ids related to the 
                # provenance semantic class id
                # Question: does this make sense?
                mapping[row.id].update(scid2senseid[p[-1]])
    return mapping


    
def obtain_quotations_for_senses(
                    df_quotations:  pd.DataFrame,
                    df_source: pd.DataFrame,
                    senses: set,
                    start:int,
                    end: int
                    ) -> pd.DataFrame:
    """Create a dataframe with quotations and their metadata for 
    a selected set of senses. This function builds on
    harvest_quotations. It also add more provenace and context 
    information of the sense via which we derive the quotations
    more precisely we add these columns:
        - "daterange": date range of sense as stated in OED
        - "provenance": sense provenance in the shape of [source, relation, target]
        - "provenance_type": sense provenance type (seed | synonym | descendant | sibling)
        - "relation_to_core_senses": via which core sense(s) (seed | synonym)
                                did we derive the quotation
        - "relation_to_seed_senses": via which seed sences did we derive the quotations
                                empty set of the quotation we derived via synonym only
    
    Arguments:
        df_quotations: dataframe with quotations, created using harvest_quotations_by_sense_id
        df_source: dataframe with additional information on senses such as provenance and daterange
                    this dataframe is created by the extend_from_lemma function
        senses (set): set of senses for which we want to obtain quotations
        start (int): start at year
        end (int): end at year
    Returns:
        pd.DataFrame with selected quotations
        
    """
    mapping = relation_to_core_senses(df_source)

    intersects_with = lambda x,target_set:  x.intersection(target_set)
    seed_ids = set(df_source[df_source.provenance_type == 'seed'].id)

    df_source['relation_to_core_senses'] = df_source.id.apply(mapping.get)
    df_source['relation_to_seed_senses'] = df_source.relation_to_core_senses.apply(intersects_with,target_set=seed_ids) 
    
    df = pd.concat([
        pd.DataFrame.from_records(df_quotations.text.values),
        pd.DataFrame.from_records(df_quotations.source.values)
            ], axis=1)
    #df['year'] = df_quotations['year']
    #df['sense_id'] = df_quotations['sense_id']
    #df['word_id'] = df_quotations['word_id']
    #df['quotation_id'] = df_quotations['quotation_id']
    df[['year','sense_id','word_id','lemma','quotation_id','definition']] = df_quotations[['year','sense_id','word_id','lemma','quotation_id','definition']]
    df = df[df.sense_id.isin(senses)]
    df = df[(start <= df.year) & (df.year <= end)]
    df.drop_duplicates(inplace=True)
    #df = df.merge(df_source[['id','daterange','definition',
    #                        "provenance","provenance_type",
    #                        "relation_to_core_senses","relation_to_seed_senses"]],
    #                        left_on='sense_id',
    #                        right_on='id',
    #                        how='left'
    #                            ).drop("id",axis=1)
    
    return df