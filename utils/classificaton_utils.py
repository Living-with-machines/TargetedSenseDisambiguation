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
from utils.dataset_download import *
from sklearn.model_selection import train_test_split
from tasks import wsd
from utils import nlp_tools
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
    df_quotations_selected = obtain_quotations_for_senses(df_quotations,
                                df_source,                  
                                senses,
                                start=start,end=end)
    #print(df_quotations_selected.columns)
    # add label column, set all labels to zero 
    df_quotations['label'] = "0"
    # set label to one for selected quotations
    df_quotations.loc[df_quotations.quotation_id.isin(df_quotations_selected.quotation_id),'label'] = "1"
    
    # strict filter is True we discard all functions outside
    # of the experiment parameters, which are defined by the
    # time period and the word types of the target senses
    if strict_filter:
        df_quotations = df_quotations[(df_quotations.lemma.isin(df_quotations_selected.lemma)) & \
                                    (df_quotations.year >= start) & \
                                    (df_quotations.year <= end) ]
                                    
    df_quotations = df_quotations.merge(df_source[['id','daterange',
                            "provenance","provenance_type",
                            "relation_to_core_senses","relation_to_seed_senses"]],
                            left_on='sense_id',
                            right_on='id',
                            how='left'
                                )#.drop("id",axis=1)
    
    if len(df_quotations)==0:
        print ("\nThere are no quotations available, given this sense-id and time-frame.")
        return None,None,None

    df_quotations.drop_duplicates(subset = ["year", "lemma", "word_id", "sense_id", "definition", "full_text"], inplace = True)
    # drop rows with vector in vector_bert_base_-1,-2,-3,-4_mean
    print('[LOG] #rows before removing None vector',df_quotations.shape)
    df_quotations = df_quotations[~df_quotations['vector_bert_base_-1,-2,-3,-4_mean'].isnull()]
    print('[LOG] #rows after removing None vector',df_quotations.shape)
    df_quotations = df_quotations.reset_index(drop=True)

    train, test = train_test_split(df_quotations, test_size=0.2, random_state=42,shuffle=True, stratify=df_quotations[['label']])
    train, val = train_test_split(train, test_size=0.2, random_state=42,shuffle=True, stratify=train[['label']])
    train = train[~train.definition.isnull()].reset_index(drop=True)
    
    if eval_mode == "lemma":
        train = train[train['lemma'] == lemma] # changed this
        train = train.reset_index(drop=True)

    
    return train,val,test

def generate_definition_df(df_train,lemma,eval_mode="lemma"):
    def merge_definitions(row):
        definition = ''
        if row.lemma_definition:
            definition += row.lemma_definition
        if row.definition:
            definition += ' & '
            definition += row.definition
        return definition

    df_selected_senses = df_train[['sense_id','lemma','word_id','lemma_definition','definition','label']]
    #df_selected_senses['definition'] = df_selected_senses.apply(merge_definitions, axis=1)


    df_selected_senses = df_selected_senses.rename(columns={'sense_id': 'id','word_id':'lemma_id'})
    df_selected_senses = df_selected_senses[~df_selected_senses.definition.isnull()]
    df_selected_senses.drop_duplicates(inplace = True)
    df_selected_senses = df_selected_senses.reset_index(drop=True)

    if eval_mode == "lemma":
        print(f'Using {eval_mode} as evaluation mode.')
        df_selected_senses = df_selected_senses[df_selected_senses['lemma'] == lemma]
        df_selected_senses = df_selected_senses.reset_index(drop=True)
        return df_selected_senses

    if eval_mode == "lemma_etal":
        print(f'Using {eval_mode} as evaluation mode.')    
        return df_selected_senses

def eval_lemma(lemma,
                pos,
                idx,
                embedding_methods,
                start=1760,
                end=1920,
                vector_type='vector_bert_base_-1,-2,-3,-4_mean',
                skip_vectorize=False,
                train_on_dev=True):

    quotations_path = f"./data/sfrel_quotations_{lemma}_{pos}.pickle"
    
    if not skip_vectorize:
        vectorize_target_expressions(quotations_path,embedding_methods)

    lemma_senses = pd.read_pickle(f'./data/lemma_senses_{lemma}_{pos}.pickle')
    senses = set(lemma_senses[lemma_senses.word_id==f'{lemma}_{pos.lower()}{idx}'].id)

    relations = ['seed','synonym'] # ,'descendant','sibling'
    eval_mode = "lemma_etal" # lemma or lemma_etal

    wemb_model = Word2Vec.load("/deezy_datadrive/kaspar-playground/dictionary_expansion/HistoricalDictionaryExpansion/models/w2v_004/w2v_words.model")
    y_true,y_pred_bin_centr, y_pred_ts_bin_centr,y_pred_sense_centr,y_pred_semaxis, rand, token_overlap,w2v_lesk = [], [],[],[], [], [],[], []


    tqdm.pandas()

    for sense in senses:
        
            print(sense)
            df_train, df_val, df_test = binarize(lemma,
                        pos,
                        {sense}, 
                        relations,
                        strict_filter=True,
                        start=start,
                        end=end,
                        eval_mode=eval_mode)
            # no quotations for sense and timeframe
            if df_train is None: continue



            y_true.extend(df_test.label.to_list())


            if train_on_dev:
                df_train = pd.concat([df_train, df_val], axis=0)

            df_train["nlp_full_text"] = df_train.apply(lambda row: nlp_tools.preprocess(row["full_text"]), axis=1)

            df_val["nlp_full_text"] = df_val.apply(lambda row: nlp_tools.preprocess(row["full_text"]), axis=1)

            df_test["nlp_full_text"] = df_test.apply(lambda row: nlp_tools.preprocess(row["full_text"]), axis=1)

            # random 
            df_test["random"] = df_test.progress_apply(lambda row: wsd.random_predict(), axis=1)
            rand.extend(df_test["random"].to_list())
            
            # token overlap
            df_selected_senses = generate_definition_df(df_train,lemma,eval_mode=eval_mode)
            df_selected_senses["nlp_definition"] = df_selected_senses.apply (lambda row: nlp_tools.preprocess(row["definition"]), axis=1)
            df_test["def_tok_overlap_ranking"] = df_test.progress_apply (lambda row: wsd.tok_overlap_ranking(row["nlp_full_text"], df_selected_senses), axis=1)
            token_overlap.extend(df_test["def_tok_overlap_ranking"].to_list())

            #w2v lesk
            # Warning: I use a Word2vec model trained on all 19thC BL corpus that is locally stored.
            df_test["w2v_lesk_ranking"] = df_test.progress_apply (lambda row: wsd.w2v_lesk_ranking(row["nlp_full_text"], df_selected_senses, wemb_model), axis=1)
            w2v_lesk.extend(df_test['w2v_lesk_ranking'].to_list())

            # binary centroid
            centroid_vectors = df_train.groupby('label')[vector_type].apply(np.mean,axis=0)
            df_test[f"bert_centroid_binary_{vector_type}"] = df_test[vector_type].progress_apply(wsd.bert_nn_centroid_vector,
                                                                                            centroid_vectors = centroid_vectors,
                                                                                            )
            y_pred_bin_centr.extend(df_test[f"bert_centroid_binary_{vector_type}"].to_list())
            #results[f"bert_centroid_binary_{vector_type}_{sense}"] = (wsd.eval(f"bert_centroid_binary_{vector_type}",df_test),len(df_test))
            
            # binary centroid time sensitive
            df_test[f"bert_ts_centroid_binary_{vector_type}"] = df_test.progress_apply(wsd.bert_nn_ts_centroid_vector, df_train=df_train, axis=1)
            y_pred_ts_bin_centr.extend(df_test[f"bert_ts_centroid_binary_{vector_type}"].to_list())
            
            #results[f"bert_ts_centroid_binary_{vector_type}_{sense}"] = (wsd.eval(f"bert_ts_centroid_binary_{vector_type}",df_test),len(df_test))
            
            # sense level centroid
            senseid2label = dict(df_test[['sense_id','label']].values)
            df_test[f"bert_centroid_sense_{vector_type}"] = df_test.progress_apply(wsd.bert_nn_sense_centroid_vector,  
                            senseid2label= senseid2label,
                            vector_col=vector_type,
                            df_train = df_train, axis=1)
            
            y_pred_sense_centr.extend(df_test[f"bert_centroid_sense_{vector_type}"].to_list())
            #results[f"bert_centroid_sense_{vector_type}_{sense}"] = (wsd.eval(f"bert_centroid_sense_{vector_type}",df_test),len(df_test))
            # semaxis
            centroid_vectors = df_train.groupby('label')[vector_type].apply(np.mean,axis=0)
            sem_axis = centroid_vectors[1] - centroid_vectors[0] 
            df_test[f"bert_semaxis_{vector_type}"] = df_test[vector_type].progress_apply(wsd.bert_semaxis_vector, sem_axis=sem_axis, return_label=True, threshold=.0)
            y_pred_semaxis.extend(df_test[f"bert_semaxis_{vector_type}"].to_list())
            #results[f"bert_semaxis_{vector_type}_{sense}"]  = (wsd.eval(f"bert_semaxis_{vector_type}",df_test),len(df_test))
    
            #df_test.to_pickle(f'./data/results/{lemma}_{pos}_{sense}.results')
    
    return y_true, y_pred_bin_centr,y_pred_ts_bin_centr,y_pred_sense_centr,y_pred_semaxis, rand, token_overlap, w2v_lesk

### --------------------------------
# Depreciated code

def bert_avg_quot_nn_wsd(query_vector: np.array,
                        quotation_df: pd.DataFrame) -> dict:
    """Function that scores the similarity of a query vector (of a target word taken from a quotations) 
    to the sense embeddings of other sense available in quotation_df. we follow the 
    procedure of (Liu et al. 2019): for each sense we average the vector representation
    and compute the cosine similarity between these sense embeddings and the query vector.
    
    Arguments:
        query_vector (np.array): vector representation of the word we want to disambiguate
        quotation_df (pd.DataFrame): dataframe with vector column.
    
    Returns:
        dictionary that maps sense_id to the cosine similarity score
    """
    # check if 
    if not hasattr(quotation_df, 'vector'):
        raise(ValueError,"""DataFrame needs a vector column containing the vector of the target word. 
            Use utils.prepare_data() to create vector for target words""")

    quotation_df_avg_by_lemma = quotation_df.groupby('sense_id')['vector'].apply(np.mean,axis=0)
    results = quotation_df_avg_by_lemma.apply(cosine_similiarity, target = query_vector).to_dict() 
    return results