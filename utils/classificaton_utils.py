import pandas as pd
import numpy as np
import flair
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings

def filter_quotations_by_year(
                      df_quotations:  pd.DataFrame,
                      start:int,
                      end: int
                    ) -> pd.DataFrame:
    """Create a dataframe with quotations and their metadata for 
    for a specific year range
    
    Arguments:
        df_quotations: dataframe with quotations, created using harvest_quotations_by_sense_id
        start (int): start year
        end (int):end year
        
    Returns:
        pd.DataFrame with quotations
        
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
                            punctuation: str = '!"#—$%&\()*+,./:;\'\—-<=>?@[\\]^_`{|}~‘’'):
    """
    Get a vector representation for a target expression in context.
    If the target expression consists of multiple words we average the 
    multiple vector representations.
    
    Arguments:
        row (pd.Series): a row from a quotations dataframe created by 
                        the function filter_quotations_by_year
    Returns:
        a np.array that captures the last layer(s) of the transformer
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
            vectors.append(token.embedding.numpy())
            quotation_target_tokens.append(token.text)
    if vectors:
        if ' '.join(quotation_target_tokens) != ' '.join(target.split()):
            print('Warning: could not properly match',' '.join(target.split()), ' with ',' '.join(quotation_target_tokens))
        
        return np.mean(vectors, axis=0)
    
    return None
