import random
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import scipy
from utils import nlp_tools

### evaluation metrics
def eval(approach,df_quotations):
    ranking = df_quotations[approach]
    gold = df_quotations["label"]
    preds = []
    for line in ranking:
        #ranking the list of list by the prediction, the higher the better
        line.sort(key=lambda x: x[1],reverse=True)
        # taking the first one as prediction (the list of list might be useful when we compute other metrics)
        p = line[0][0]
        preds.append(p)
    # we report p,r,f1 for both labels
    p_1,r_1,f1_1 = [round(x,3) for x in precision_recall_fscore_support(gold,preds, average='binary',pos_label="1")[:3]]
    p_0,r_0,f1_0 = [round(x,3) for x in precision_recall_fscore_support(gold,preds, average='binary',pos_label="0")[:3]]
    results = {"1":[p_1,r_1,f1_1 ],"0":[p_0,r_0,f1_0]}
    return results

### random baseline
def random_predict(definition_df):
    definition_df["random"] = definition_df.apply (lambda row: str(random.randint(0, 1)), axis=1)
    results = definition_df[['label','random']].values.tolist()
    return results

### token overlap baseline
def tok_overlap_ranking(sent,definition_df):
    definition_df["tok_overlap"] = definition_df.apply(lambda row: token_overlap(sent,row["nlp_definition"]), axis=1)
    results = definition_df[['label','tok_overlap']].values.tolist()
    return results

def token_overlap(sent1,sent2):
    sent1 = set([tok.lemma_ for tok in sent1 if not tok.is_punct and not tok.is_stop])
    sent2 = set([tok.lemma_ for tok in sent2 if not tok.is_punct and not tok.is_stop])
    score = len(sent1 & sent2)
    return score

# sentence embedding similarity
def sent_embedding(sent,definition_df):
    definition_df["sent_embedding"] = definition_df.apply (lambda row: sent.similarity(row["nlp_definition"]), axis=1)
    results = definition_df[['label','sent_embedding']].values.tolist()
    return results


### ---------------------------------------------------
### Word2Vec Lesk WSD baseline
    
def w2v_lesk_wsd(sent1, sent2, wemb_model):
    sent1 = [tok.lemma_ for tok in sent1 if not tok.is_punct and not tok.is_stop]
    sent2 = [tok.lemma_ for tok in sent2 if not tok.is_punct and not tok.is_stop]
    
    sent1_embedding = nlp_tools.avg_embedding(sent1,wemb_model).reshape(1,-1)
    sent2_embedding = nlp_tools.avg_embedding(sent2,wemb_model).reshape(1,-1)
    
    sim = 1.0 - scipy.spatial.distance.cdist(sent1_embedding, sent2_embedding, "cosine")[0][0]
    return sim

def w2v_lesk_ranking(sent, definition_df, wemb_model):
    definition_df["w2v_lesk_ranking"] = definition_df.apply(lambda row: w2v_lesk_wsd(sent, row["nlp_definition"], wemb_model), axis=1)
    results = definition_df[['label','w2v_lesk_ranking']].values.tolist()
    return results


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
    return results