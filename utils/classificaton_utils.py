from numpy.core.numeric import outer
from numpy.lib.financial import ppmt
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings
from scipy.spatial.distance import cosine
from pathlib import Path, PosixPath
from typing import Union
from collections import defaultdict
from utils.dataset_download import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import ttest_ind

#import swifter

cosine_similiarity = lambda x, target : 1 - cosine(x,target)

def filter_quotations_by_year(
                    df_quotations: pd.DataFrame,
                    start: int,
                    end: int
                    ) -> pd.DataFrame:
    """Create a dataframe with quotations and their metadata for 
    for a specific year range
    
    Arguments:
        df_quotations: dataframe with quotations, created using harvest_quotations_by_sense_id
        start (int): start year
        end (int):end year
        
    Returns:
        pd.DataFrame with quotations, the dataframe contains year, sense_id, and word_id as columnss
        
    """
    df = pd.concat([
        pd.DataFrame.from_records(df_quotations.text.values),
        pd.DataFrame.from_records(df_quotations.source.values)
            ], axis=1)
    df['year'] = df_quotations['year']
    df['sense_id'] = df_quotations['sense_id']
    df['word_id'] = df_quotations['word_id']
    #df = df[df.sense_id.isin(senses)]
    df = df[(start <= df.year) & (df.year <= end)]
    
    df.drop_duplicates(inplace=True)
    
    return df

def get_target_token_vector(row: pd.Series, 
                            embedding_type: TransformerWordEmbeddings,
                            punctuation: str = '!"#—$%&\()*+,./:;\'\—-<=>?@[\\]^_`{|}~‘’',
                            combine:Union[str,None]='average') -> Union[np.array,list]:
    """
    Get a vector representation for a target expression in context.
    If the target expression consists of multiple words we average the 
    multiple vector representations. The function assumes a quotations 
    from the OED as input, which has the following values:
        - keyword: the target word
        - full_text: the quotation
        - keyword_offset: the first character of the target word
    
    Arguments:
        row (pd.Series): a row from a quotations dataframe created by 
                        the function filter_quotations_by_year
        embedding_type (TransformerWordEmbeddings):
        punctuation (str): a string with all the punctuation used for tokenising words, 
                        i.e. element in this string will be replace with a white space
        combine ('average', None): specify strategy for combining vectors 
                        if the target word consists of multiple tokens
    Returns:
        a np.array that captures the last layer(s) of the transformer or a list with vectors
    """
    # replace all punctuation with white spaces
    text = ''.join([' ' if c in punctuation else c  for c in row.full_text.lower()])
    
    # if there is no quotation return None
    if text is '':
        return None
    
    text = Sentence(text,use_tokenizer=False)
    target = row.keyword # the offset as recorded by the OED
    vectors = []; quotation_target_tokens = [] # we collect the target tokens collected in the quotation
                                                # and match those with the target expression as a check (see below)
    
    # if there is no target word return none
    # remove punctuation from target expression
    if target is not None:
        target = ''.join([' ' if c in punctuation else c  for c in target.lower()])
    else:
        return None
    
    # get offsets of the target expression in the quotations
    start_position = row.keyword_offset
    end_position = start_position + len(target)
    
    # embedd text
    embedding_type.embed(text)
    
    for token in text:
        # here we rely on the FLAIR offset annotation in combination with tokenisation
        # double check if this works properly
        if (token.start_pos >= start_position) and (token.start_pos < end_position):
            # when using CUDA move to cpu
            vectors.append(token.embedding.cpu().numpy())
            quotation_target_tokens.append(token.text)

    if vectors:
        if ' '.join(quotation_target_tokens) != ' '.join(target.split()):
            print('[WARNING] Could not properly match',' '.join(target.split()), ' with ',' '.join(quotation_target_tokens), " Return None.")
            return None
        if combine == 'average':
            return np.mean(vectors, axis=0)
        if combine is None:
            # return the vectors as a list
            # TO DO: add other functions for combining the vectors for the target word
            return vectors
        else:
            print(f"Method: {combine} for combining vectors is not implemented. Return None.")
            return None
    else:
        print("[WARNING] 'vectors' variable is empty. Return None.")
        return None

def vectorize_target_expressions(
                quotations_path: PosixPath, 
                embedding_method: dict,
                ) -> pd.DataFrame:
    """prepare data for word sense disambiguation with quotations
    this function vectorizes target words in a quotation (meaning
    we have a vector representation for the quotation keyword)
    the vector representations are saved in the dataframe in the
    `vector_{bert_name}_{settings}` column. 
    
    `embedding_method` is a dictionary that contains the name of 
    the BERT model and the specific settings
    for extracting contextual vector representation (i.e. the number of
    layers and the pooling_operation)

    Arguments:
        path (PoxixPath): path to dataframe with all sense ids and quotations
        embedding_method (dict): dictionary with { column_name : 
                                                    { path: path_to_bert_model, 
                                                        layers=-1, 
                                                        pooling_operation='mean'
                                                        }}

    Returns:
        a pandas.DataFrame with quotations that or filtered by time
        and which are processed for sense disambiguation using the vector
        representation of the target word (or keyword)
    """
    
    quotations = pd.read_pickle(quotations_path)

    for bert_name, bert_settings in embedding_method.items():
        layers = layers=bert_settings['layers']
        pooling_operation = bert_settings['pooling_operation']
        col_name = f'vector_{bert_name}_{layers}_{pooling_operation}'
        
        # if dataframe already contains column, skip
        if hasattr(quotations, col_name):
            print(f'Dataframe alread contains vectors from {bert_name} settings')
            print(bert_settings)

            continue
        
        # load embedding model
        embedding_type = TransformerWordEmbeddings(
                                bert_settings['path'],
                                layers=layers,
                                pooling_operation=pooling_operation)

        # embded
        quotations[col_name] = quotations.progress_apply(get_target_token_vector,
                                                        embedding_type=embedding_type,
                                                        axis=1)

    quotations.to_pickle(quotations_path)
    return quotations

def binarize(lemma:str,
            pos: str,
            senses:set,
            relations:list,
            expand_seeds:bool=True,
            expand_synonyms:bool=True,
            start:int=1760, 
            end:int=1920,
            strict_filter:bool=True,
            filter_test_by_year=True,
            filter_val_by_year=True,
            eval_mode="lemma") -> pd.DataFrame:
    """binarize labels and select quotations
    given a set of senses, provenance rules, and expansion flags,
    this function selects all relevant, related senses, and obtains quotations 
    that fall within the specified target period. 

    This function requires dataframe created by
        - extend_from_lemma
        - harvest_quotations
        (Use pipeline.py to create these dataframes for a specific lemma id)

    The strict_filter arguments will discard any quotation that is
    outside the time period and has a different word id (compared to
    the senses retrieved via the filter_sensen function)

    Arguments:
        lemma_id (str):
        senses (set):
        relations (list):
        filter_type (strict,loose): retain or discard items don't match the parameters
        eval_mode (lemma,lemma_etal): determines the scope of the training set. if set 
                                    to lemma, use only sense that are directly part of
                                    the original lemma, if set to lemma_etal use lemma
                                    and the expanded set of senses. 
    """
    # load core dataset for a given lemma_id
    df_source = pd.read_pickle(f'./data/extended_senses_{lemma}_{pos}.pickle')
    df_quotations = pd.read_pickle(f'./data/sfrel_quotations_{lemma}_{pos}.pickle')

    #print(df_quotations.columns)
    # filter senses
    senses = filter_senses(df_source,
                    senses,
                    relations = relations,  
                    expand_seeds=expand_seeds,
                    expand_synonyms=expand_synonyms,
                    start=start, 
                    end=end
                    )
    
    # get the quotations for the filtered senses
    # !! important change here: the start and end flag
    # only effect the filtering of the senses not the 
    # quotations themselves, we only filter quotations
    # by time for the test set
    df_quotations_selected = obtain_quotations_for_senses(df_quotations,
                                df_source,                  
                                senses,
                                start=0,end=2021)
    #print(df_quotations_selected.columns)
    # add label column, set all labels to zero 
    df_quotations['label'] = "0"
    # set label to one for selected quotations
    df_quotations.loc[df_quotations.quotation_id.isin(df_quotations_selected.quotation_id),'label'] = "1"
    
    # strict filter is True we discard all functions outside
    # of the experiment parameters
    # we first filter out all lemma which are not related
    # (via their surface form) to the selected senses
    # we keep, for now, quotations from all years
    if strict_filter:
        df_quotations = df_quotations[(df_quotations.lemma.isin(df_quotations_selected.lemma))] #& \
                                    #(df_quotations.year >= start) & \
                                    #(df_quotations.year <= end) 
                                    #]

    if len(df_quotations)==0:
        print ("\nThere are no quotations available, given this sense-id and time-frame.")
        return None,None,None

    df_quotations = df_quotations.merge(df_source[['id','daterange',
                            "provenance","provenance_type",
                            "relation_to_core_senses","relation_to_seed_senses"]],
                            left_on='sense_id',
                            right_on='id',
                            how='left'
                                )#.drop("id",axis=1)
    
    df_quotations.drop_duplicates(subset = ["year", "lemma", "word_id", "sense_id", "definition", "full_text"], inplace = True)
    # drop observation that can't be used 
    # drop rows with vector in vector_bert_base_-1,-2,-3,-4_mean
    df_quotations = df_quotations[~df_quotations['vector_bert_base_-1,-2,-3,-4_mean'].isnull()]
    
    # drop observations which are not ambiguous
    # counts the number of senses per lemma
    sense_count_by_lemma = df_quotations.groupby('lemma')['sense_id'].unique().apply(len)
    # ambiguous lemmas are those with more than one sense
    ambiguous_lemmas = list(sense_count_by_lemma[sense_count_by_lemma > 1].index)
    # retain only ambiguouse lemmas
    df_quotations = df_quotations[df_quotations.lemma.isin(ambiguous_lemmas)]
    df_quotations = df_quotations.reset_index(drop=True)
    
    train, test = train_test_split(df_quotations, test_size=0.25, random_state=42,shuffle=True, stratify=df_quotations[['label']])
    
    # ! Important: we want to evaluate only on quotations for the test set that fit
    # the date range defined by start end end
    if filter_test_by_year:
        test = test[(start <= test.year) & (test.year <= end)]
    if len(test)==0:
        print ("\nThere are no quotations available in the test set, given this sense-id and time-frame.")
        return None,None,None

    train, val = train_test_split(train, test_size=0.2, random_state=42,shuffle=True, stratify=train[['label']])
    
    if filter_val_by_year:
        val = val[(start <= val.year) & (val.year <= end)]
    
    train = train[~train.definition.isnull()].reset_index(drop=True)
    print(f"[LOG] {train.shape[0] + val.shape[0] + test.shape[0]} quotations selected")
    print(f"[LOG] train = {train.shape[0]} val = {val.shape[0]} test = {test.shape[0]} quotations")
    if eval_mode == "lemma":
        train = train[train['lemma'] == lemma] # changed this
        train = train.reset_index(drop=True)

    
    return train,val,test

def generate_definition_df(df_train,lemma,eval_mode="lemma"):
    df_selected_senses = df_train[['sense_id','lemma','word_id','lemma_definition','definition','label']]


    df_selected_senses = df_selected_senses.rename(columns={'sense_id': 'id','word_id':'lemma_id'})
    df_selected_senses = df_selected_senses[~df_selected_senses.definition.isnull()]
    df_selected_senses.drop_duplicates(inplace = True)
    df_selected_senses = df_selected_senses.reset_index(drop=True)

    if eval_mode == "lemma":
        print(f'[LOG] Using {eval_mode} as evaluation mode.')
        df_selected_senses = df_selected_senses[df_selected_senses['lemma'] == lemma]
        df_selected_senses = df_selected_senses.reset_index(drop=True)
        return df_selected_senses

    if eval_mode == "lemma_etal":
        print(f'[LOG] Using {eval_mode} as evaluation mode.')    
        return df_selected_senses

def evaluate_results(results_path):
    clf_dict = defaultdict(list)
    results = {}
    csv_files = results_path.glob("**/*.csv")
    for csv in csv_files:
        try:
            df = pd.read_csv(csv)

        except Exception as e:
            #print(e)
            #print(csv)
            continue
        for col in df.columns:
            clf_dict[col].extend(df[col])
    
    for colname, classifications in clf_dict.items():
        if colname not in ['label','year','quotation_id']:
            results[colname] =  [round(x,3) for x in precision_recall_fscore_support(clf_dict['label'],classifications,average='binary',pos_label=1) if x] + [[classifications]]
    return results

