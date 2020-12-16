import scipy
import random
import numpy as np
from sklearn import svm
from utils import nlp_tools
from sklearn.metrics import precision_recall_fscore_support

### evaluation metrics
def eval(approach,df_quotations):
    gold = df_quotations["label"]
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

def svm_wemb_baseline(df_train,X_test,wemb_model):

    df_train["sent_emb"] = df_train.apply(lambda row: nlp_tools.avg_embedding(row["nlp_full_text"],wemb_model).reshape(1,-1)[0], axis=1)
    X_test = nlp_tools.avg_embedding(X_test,wemb_model).reshape(1,-1)[0]

    X_train = np.array(df_train["sent_emb"].tolist())
    y_train = np.array(df_train["label"].tolist())

    X_test = np.array([X_test])

    classifier = SVM.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)[0]

    return y_pred