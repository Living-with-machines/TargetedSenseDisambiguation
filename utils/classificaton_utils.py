from numpy.core.numeric import outer
from numpy.lib.financial import ppmt
import pandas as pd
import numpy as np
import flair
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings
from scipy.spatial.distance import cosine
from pathlib import Path, PosixPath
from typing import Union

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

def prepare_data(path: PosixPath, 
                embedding_type: TransformerWordEmbeddings,
                start_year:int=1760, 
                end_year:int=1920) -> pd.DataFrame:
    """prepare data for word sense disambiguation with quotations
    this function filters quotations for a given date range
    it then checks if all target words have been vectorized (mean
    we have a vector representation for the quotation keyword)
    if not, we add a `vector` column to dataframe and save it.
    Arguments:
        path (PoxixPath): path to dataframe with all sense ids and quotations
        embedding_type (TransformerWordEmbeddings): Transformer used for generating
            the vector representation used for disambiguation
        start_year (int): start filter at year
        end_year (int): end filter at year

    Returns:
        a pandas.DataFrame with quotations that or filtered by time
        and which are processed for sense disambiguation using the vector
        representation of the target word (or keyword)
    """
    data = pd.read_pickle(path)
    
    quotations_path = path.parent / f"{path.stem}_{start_year}_{end_year}.pickle"
    
    if not quotations_path.is_file():
        print(f'Quotations file: {quotations_path} could not be found. Vectorizing the target word...')
        quotations = filter_quotations_by_year(data,start=start_year,end=end_year)
        try:
            import swifter
            print("[INFO] swifter is installed. Parallelize pandas apply method.")
            quotations['vector'] = quotations.swifter.apply(get_target_token_vector,
                                                            embedding_type=embedding_type,
                                                            axis=1)
        except ImportError:
            print("[WARNING] could not find swifter...run pandas apply on one process.")
            quotations['vector'] = quotations.apply(get_target_token_vector,
                                                    embedding_type=embedding_type,
                                                    axis=1)
        quotations.to_pickle(quotations_path)
        print("Done. Created dataframe with vectors for target words.")
        print(f'Saved Dataframe: {quotations_path}')
    else:
        quotations = pd.read_pickle(quotations_path)
    
    return quotations

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