import os
import scipy
import random
import pandas as pd
import numpy as np
import pandas as pd
from sklearn import svm
from utils import nlp_tools
from typing import Union
from sklearn.metrics import precision_recall_fscore_support
from utils.classificaton_utils import cosine_similiarity

### evaluation metrics
def compute_eval_metrics(word,results_path,eval_mode):
    micro = {}
    true = []
    # just to avoid picking up annoying folders
    if not word.startswith('.'):
        experiments = os.path.join(results_path, word, eval_mode)

        # for each experiment, so basically for each sense
        for exp_file in os.listdir(experiments):
            if exp_file.endswith('.csv'):
                exp_path = os.path.join(experiments, exp_file)
                try:
                    experiment = pd.read_csv(exp_path, sep=",")
                    label = experiment["label"].tolist()
                    true+=label

                    methods = experiment.columns.tolist()
                    methods.remove("label")

                    for method in methods:
                        pred = experiment[method].tolist()

                        if method in micro:
                            micro[method]+=pred
                        else:
                            micro[method] = pred

                # we have some empty files <-- to be checked
                except pd.io.common.EmptyDataError:
                    print ("\t --> No results stored for", exp_path)

        columns = ["method","p","r","f1"]
        res_df = []

        for method,micro_preds in micro.items():
            row = [method]
            p,r,f1 = [round(x,3) for x in precision_recall_fscore_support(true,micro_preds, average='binary',pos_label=1)[:3]]
            row+=[p,r,f1]
            res_df.append(row)

        res_df = pd.DataFrame(res_df,columns=columns)
        return res_df

### ---------------------------------------------------
### random baseline
def random_predict():
    y_pred = str(random.randint(0, 1))
    return y_pred

### ---------------------------------------------------
### token overlap baseline
def tok_overlap_ranking(sent,definition_df):
    definition_df["tok_overlap"] = definition_df.apply(lambda row: token_overlap(sent,row["nlp_definition"]), axis=1)
    results = definition_df[['label','tok_overlap']].values.tolist()
    results.sort(key=lambda x: x[1],reverse=True)
    y_pred = results[0][0]
    return y_pred

def token_overlap(sent1,sent2):
    sent1 = set([tok.lemma_ for tok in sent1 if not tok.is_punct and not tok.is_stop])
    sent2 = set([tok.lemma_ for tok in sent2 if not tok.is_punct and not tok.is_stop])
    score = len(sent1 & sent2)
    return score

### ---------------------------------------------------
# sentence embedding similarity
def sent_embedding(sent,definition_df):
    definition_df["sent_embedding"] = definition_df.apply (lambda row: sent.similarity(row["nlp_definition"]), axis=1)
    results = definition_df[['label','sent_embedding']].values.tolist()
    results.sort(key=lambda x: x[1],reverse=True)
    y_pred = results[0][0]
    return y_pred


### ---------------------------------------------------
### Word2Vec Lesk WSD baseline
    
def w2v_lesk_wsd(sent1, sent2, wemb_model):

    sent1_embedding = nlp_tools.avg_embedding(sent1,wemb_model).reshape(1,-1)
    sent2_embedding = nlp_tools.avg_embedding(sent2,wemb_model).reshape(1,-1)
    
    sim = 1.0 - scipy.spatial.distance.cdist(sent1_embedding, sent2_embedding, "cosine")[0][0]
    return sim

def w2v_lesk_ranking(sent, definition_df, wemb_model):
    definition_df["w2v_lesk_ranking"] = definition_df.apply(lambda row: w2v_lesk_wsd(sent, row["nlp_definition"], wemb_model), axis=1)
    results = definition_df[['label','w2v_lesk_ranking']].values.tolist()
    results.sort(key=lambda x: x[1],reverse=True)
    y_pred = results[0][0]
    return y_pred


### ---------------------------------------------------
### BERT Lesk WSD baseline
    
def bert_lesk_wsd(sent1, sent2, bert_sentsim_model):
    
    sent1_embedding = bert_sentsim_model.encode([sent1]) # Full sentence
    sent2_embedding = bert_sentsim_model.encode([sent2]) # Full sentence
    
    sim = 1.0 - scipy.spatial.distance.cdist(sent1_embedding, sent2_embedding, "cosine")[0][0]
    return sim

def bert_lesk_ranking(sent, definition_df, bert_sentsim_model):
    definition_df["bert_lesk_ranking"] = definition_df.apply(lambda row: bert_lesk_wsd(sent, row["definition"], bert_sentsim_model), axis=1)
    results = definition_df[['label','bert_lesk_ranking']].values.tolist()
    results.sort(key=lambda x: x[1],reverse=True)
    y_pred = results[0][0]
    return y_pred

### ---------------------------------------------------
# SVM word embedding baseline
SVM = svm.SVC(kernel = "linear", C=1, probability=True)

def svm_wemb_baseline(df_train,df_test,wemb_model):

    df_train["sent_emb"] = df_train.apply(lambda row: nlp_tools.avg_embedding(row["nlp_full_text"],wemb_model).reshape(1,-1)[0], axis=1)
    df_test["sent_emb"] = df_test.apply(lambda row: nlp_tools.avg_embedding(row["nlp_full_text"],wemb_model).reshape(1,-1)[0], axis=1)

    X_train = np.array(df_train["sent_emb"].tolist())
    y_train = np.array(df_train["label"].tolist())

    X_test = np.array(df_test["sent_emb"].tolist())

    classifier = SVM.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)

    return y_pred

### ---------------------------------------------------
# BERT CENTROID METHODS

### ---------------------------------------------------
# binary centroid vectors

def bert_binary_centroid_vector(row:pd.Series,
                            df_train:pd.DataFrame,
                            return_ranking=False,
                            vector_col='vector_bert_base_-1,-2,-3,-4_mean') -> str:
    """BERT wsd disambiguation method using a centroid vectors
    representing the positive and the negative class. the class 
    is the nearest centroid. centroids are computed by averaging
    the 

    Arguments:
        row (pd.Sries): row of the test dataframe on which of the function operates
        df_train (np.Series): dataframe with training data
        return_rank (bool): if True return return scores as a dict
        vector_col (str): name of the column in which the target vector is stored

    Returns:
        class as "0" or "1" string
    """
    centroid_vectors = df_train.groupby('label')[vector_col].apply(np.mean,axis=0)

    sims = centroid_vectors.apply(cosine_similiarity, target = row[vector_col])
    
    if return_ranking:
        return sims.to_dict()
    
    return str(np.argmax(sims))

### ---------------------------------------------------
# sense level centroid vectors

def bert_sense_centroid_vector(row:pd.Series,
                                df_train:pd.DataFrame,
                                senseid2label:dict,
                                return_ranking:bool=False,
                                vector_col:str='vector_bert_base_-1,-2,-3,-4_mean') -> str:

    """BERT wsd disambiguation method using centroid vectors
    at the sense level. The function agregates vectors by sense_id.
    The prediction is the label of the close sense vector.

    Arguments:
        row (pd.Sries): row of the test dataframe on which of the function operates
        df_train (np.Series): dataframe with training data
        senseid2label (dict): dictionary that maps sense_id to a binary label
        return_rank (bool): if True return return scores as a dict
        vector_col (str): name of the column in which the target vector is stored

    Returns:
        class as "0" or "1" string
    """
    
    # what if the lemma only has one sense, include exception here
    df_train_lemma = df_train[df_train.lemma==row.lemma]
    
    # if lemma doesn't appear in train return '0'
    if not df_train_lemma.shape[0]: return '0'
    
    sense_centroid_vectors = df_train_lemma.groupby('sense_id')[vector_col].apply(np.mean,axis=0)

    sims = sense_centroid_vectors.apply(cosine_similiarity, target = row[vector_col]).to_dict()
    if return_ranking:
        return sims
    # there was a KeyError here, avoided it with `.get()` but check later what happened
    return senseid2label.get(sorted(sims.items(),key=lambda x: x[1], reverse=True)[0][0],"0")

### ---------------------------------------------------
# Time-sensitive methods

### ---------------------------------------------------
# helper functions for creating time sensisitve sense vectors

def weighted(df,year,vector_col,level='label') -> pd.Series:
    """This function weights vector representation of 
    target words by their distance to the year
    of the query vector. This is repeated for each 
    sense_id or label (i.e. value of `level` argument). 

    It returns sense level or binary time weighted centroid vectors

    Arguments:
        df (pd.DataFrame): the training data from which to construct
                        the time sensitive embedding
        year (int): year of the vector to disambiguate
        vector_col (str): name of the column in which the target vector is stored
        level (str): use 'label' for binary centroid vector, 
                    use `sense_id` for sense level centroid vectors

    Returns:
        as element of type pd.Series with index=level and 
        values the centroid vector (in this the weighted vectors
        averaged by the specified level)

    """
    # 1 over the distance in years
    df['temp_dist'] = (1 / (abs(year - df.year) + 1))
    # normalize, so weights add up to one
    df['temp_dist'] = df['temp_dist'] / sum(df['temp_dist'])
    # time weighted vector (tw_vector) is the product of the vector and the weight
    df['tw_vector'] = df[vector_col] * df['temp_dist']
    # sum vectors by label (sum or mean??)
    return df.groupby(level)['tw_vector'].apply(np.sum,axis=0)          

def nearest(df:pd.DataFrame,
            year:int,
            vector_col:str,
            level:str='label') -> pd.Series:
    """This function selects the quotation closest in time 
    to`year` this for each sense_id or label (i.e. value of `level` argument)

    Arguments:
        df (pd.DataFrame): the training data from which to construct
                        the time sensitive embedding
        year (int): year of the vector to disambiguate
        vector_col (str): name of the colums in which vector is stord
        level (str): use 'label' for binary centroid vector, 
                    use `sense_id` for sense level centroid vectors

    Returns:
        as element of type pd.Series with index=level and 
        values the centroid vector (in this case the vector
        closest in time for the specified level)
    """
    # this methods obtains the quotation closest in time for each sense of a lemma. 
    # get idx of quotations nearest in time for each sense
    df['temp_dist'] = abs(df.year - year)
    quots_nn_time_idx = df.groupby(level)['temp_dist'].idxmin().values
    # get the quotations and the sense idx
    return df.loc[quots_nn_time_idx][[level,vector_col]].set_index(level,inplace=False)[vector_col]

### ---------------------------------------------------
# time-sensitive centoid disambiguation functions
# time sensitive binary centroid vectors

def bert_ts_binary_centroid_vector(row:pd.Series,
                            df_train:pd.DataFrame,
                            ts_method:str='weighted',
                            return_ranking:bool=False,
                            vector_col:str='vector_bert_base_-1,-2,-3,-4_mean') -> str:
    """time-sensitive wsd disambiguation method using a centroid vectors
    for the positive and negative class. the nearest of the centroid vectors
    determines the class.

    Arguments:
        row (pd.Series): row of df_test to which method is applied
        df_train (pd.DataFrame): training data used for creating centroids
        ts_method (str): specify options for time sensitive weighting 
                        ['weighted','nearest']
        return_rank (bool): if True return return scores as a dict
        vector_col (str): columns used for computing centroids
        
    Returns:
        class as "0" or "1" as string
    """

    vector, year = row[vector_col],row.year     
    
    ts_methods = ['weighted','nearest']
    

    if ts_method=='weighted':
        # weight vector by distance
        centroid_vectors = weighted(df_train,year,vector_col)
    elif ts_method=='nearest':
        # the nearest vector in time
        centroid_vectors = nearest(df_train,year,vector_col)
    else:
        assert ts_method in ts_methods, f'ts_method should be one of the following options {ts_methods}'

    sims = centroid_vectors.apply(cosine_similiarity, target = vector)
    
    if return_ranking:
        return sims.to_dict()
    return str(np.argmax(sims))

### ---------------------------------------------------
# time-sensitive sense level centroid vectors

def bert_ts_sense_centroid_vector(row:pd.Series,
                                df_train:pd.DataFrame,
                                senseid2label:dict,
                                ts_method:str='nearest',
                                return_ranking:bool=False,
                                vector_col:str='vector_bert_base_-1,-2,-3,-4_mean') -> str:

    """bert wsd disambiguation method using a centroid vectors
    at the sense level. the time-sensitive
    sense embedding is vector of the keyword of the quotation
    nearest in time.

    Arguments:
        row (pd.Series): row of df_test dataframe
        df_train (pd.DataFrame): dataframe with training data
        senseid2label (dict): mapping of sense idx to binary label
        ts_method (str): specify options for time sensitive weighting 
                        ['nearest','weighted']
        return_rank (bool): if True return return scores as a dict
        vector_col (str): name of vector column 

    Returns:
        class as "0" or "1" string
    """
    
    # what if the lemma only has one sense, include exception here
    df_train_lemma = df_train[df_train.lemma==row.lemma]

    # if lemma doesn't appear in train return '0'
    if not df_train_lemma.shape[0]: return '0'

    ts_methods = ['nearest','weighted']

    if ts_method=='weighted':
        # weight vector by distance
        centroid_vectors = weighted(df_train_lemma,row.year,vector_col,level='sense_id')
    elif ts_method=='nearest':
        # the nearest vector in time
        centroid_vectors = nearest(df_train_lemma,row.year,vector_col,level='sense_id')
    else:
        assert ts_method in ts_methods, f'ts_method should be one of the following options {ts_methods}'

    #centroid_vectors
    sims = centroid_vectors.apply(
                    cosine_similiarity, target = row[vector_col]
                        ).to_dict()

    if return_ranking:
        return sims
    
    # there was a KeyError here, avoided it with `.get()` but check later what happened
    # return label as '0' or '1'
    return senseid2label.get(
            sorted(
                sims.items(),
                        key=lambda x: x[1], reverse=True)[0][0],'0'
                        )

### ---------------------------------------------------
# SEMAXIS
# bert disambiguation with contrastive semantic axis
def bert_semaxis_vector(vector:np.array,
                    sem_axis:np.array,
                    threshold:float=.5,
                    return_ranking=False) -> Union[str,float]:
    """bert wsd disambiguation method using the intuition
    behind the semaxis paper. we project the vector
    on the semantic axi

    Arguments:
        vector (np.array): vector representation of keyword to be disambiguated
        sem_axis (np.Series): semantic axis obtain by substracting the aggregated
                            vector for the positive class with the aggregated 
                            vector of the negative class
        return_ranking (bool): flag that is set to False will return the similarity
                        score and not the label, this is used to get a threshold
                        value based on the development set
    Returns:
        class as "0" or "1" string or similarity score as float
    """
    similary = cosine_similiarity(vector,sem_axis)

    if return_ranking: return similary
    
    if similary > threshold:
        return "1"
    return "0"