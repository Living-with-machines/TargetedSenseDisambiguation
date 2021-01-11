import scipy
import random
import numpy as np
import pandas as pd
from sklearn import svm
from utils import nlp_tools
from typing import Union
from sklearn.metrics import precision_recall_fscore_support
from utils.classificaton_utils import cosine_similiarity

### evaluation metrics
def eval(approach,df_quotations):
    gold = df_quotations["label"]
    preds = df_quotations[approach]
    # we report p,r,f1 for both labels
    p_1,r_1,f1_1 = [round(x,3) for x in precision_recall_fscore_support(gold,preds, average='binary',pos_label="1")[:3]]
    p_0,r_0,f1_0 = [round(x,3) for x in precision_recall_fscore_support(gold,preds, average='binary',pos_label="0")[:3]]
    results = {"1":[p_1,r_1,f1_1 ],"0":[p_0,r_0,f1_0]}
    return results

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
# bert disambiguation with centroids

def bert_nn_centroid_vector(vector:np.array,centroid_vectors:pd.Series) -> str:
    """bert wsd disambiguation method using a centroid vectors
    representing the positive and the negative class. the class 
    is the nearest centroid. centroids are computed by averaging
    the 

    Arguments:
        vector (np.array): vector representation of keyword to be disambiguated
        polar_vectors (np.Series): series with two vectors, 1 representing the 
                    positive class, 0 representing the negative class
    Returns:
        class as "0" or "1" string
    """
    return str(np.argmax(centroid_vectors.apply(cosine_similiarity, target = vector)))

# bert disambiguation with contrastive semantic axis
def bert_semaxis_vector(vector:np.array,sem_axis:np.array, threshold:float=.5, return_label=True) -> Union[str,float]:
    """bert wsd disambiguation method using the intuition
    behind the semaxis paper. we project the vector
    on the semantic axi

    Arguments:
        vector (np.array): vector representation of keyword to be disambiguated
        sem_axis (np.Series): semantic axis obtain by substracting the aggregated
                            vector for the positive class with the aggregated 
                            vector of the negative class
        return_label (bool): flag that is set to False will return the similarity
                        score and not the label, this is used to get a threshold
                        value based on the development set
    Returns:
        class as "0" or "1" string or similarity score as float
    """
    similary = cosine_similiarity(vector,sem_axis)

    if not return_label: return similary
    
    if similary > threshold:
        return "1"
    return "0"

def bert_nn_ts_centroid_vector(row:pd.Series,
                            df_train:pd.DataFrame,
                            vector_col:str='vector_bert_base_-1,-2,-3,-4_mean') -> str:
    """time-sensitive wsd disambiguation method using a polar vectors
    representing the positive and negative class. the class 
    is the nearest of the two polar vectors. the contr

    ...

    Arguments:
        row (pd.Series): row to which method is applied
        df_train (pd.DataFrame): training data used for creating centroids
        vector_col (str): columns used for computing centroids
        
    Returns:
        class as "0" or "1" as string
    """

    vector, year = row[vector_col],row.year
    
    df_train['temp_dist'] = (1 / (abs(year - df_train.year) + 1))
    df_train['temp_dist'] = df_train['temp_dist'] / sum(df_train['temp_dist'])
    df_train['tw_vector'] = df_train[vector_col] * df_train['temp_dist']
    centroid_vectors = df_train.groupby('label')['tw_vector'].apply(np.mean,axis=0)

    return str(np.argmax(centroid_vectors.apply(cosine_similiarity, target = vector)))

def bert_ts_semaxis_vector(row:pd.Series,
                        df_train:pd.DataFrame,
                        vector_col:str='vector_bert_base_-1,-2,-3,-4_mean',
                        threshold=.0,
                        return_label=True
                            ) -> str:
    """time-sensitive wsd disambiguation method using a semaxis vector.
    ...

    Arguments:
        row (pd.Series): row to which method is applied
        df_train (pd.DataFrame): training data used for creating centroids
        vector_col (str): columns used for computing centroids
        ...


    Returns:
        class as "0" or "1" as string
    """

    vector, year = row[vector_col],row.year
    
    df_train['temp_dist'] = (1 / (abs(year - df_train.year) + 1))
    df_train['temp_dist'] = df_train['temp_dist'] / sum(df_train['temp_dist'])
    df_train['tw_vector'] = df_train[vector_col] * df_train['temp_dist']
    centroid_vectors = df_train.groupby('label')['tw_vector'].apply(np.mean,axis=0)
    semaxis_vector = centroid_vectors[1] - centroid_vectors[0]
    similary = cosine_similiarity(vector,semaxis_vector)

    if not return_label: return similary
    
    if similary > threshold:
        return "1"
    return "0"