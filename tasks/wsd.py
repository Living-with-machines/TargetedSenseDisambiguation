from random import shuffle
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import scipy
from utils import nlp_tools

### evaluation metrics
def eval(ranking,gold):
    preds = []
    for line in ranking:
        #ranking the dictionary by the label, the higher the better
        sort_ranking = [[sense_id, score] for sense_id, score in sorted(line.items(), key=lambda item: item[1],reverse=True)]
        # taking the first one as prediction (the list of list might be useful when we compute other metrics)
        p = sort_ranking[0][0]
        preds.append(p)
    p,r,f1 = [round(x,3) for x in precision_recall_fscore_support(gold,preds, average='macro')[:3]]
    microf1 = round(precision_recall_fscore_support(gold,preds, average='micro')[2],3)
    return p,r,f1,microf1

### random baseline
def random_predict(definition_df):
    sense_ids = definition_df["sense_id"].tolist()
    shuffle(sense_ids)
    results = {sense_ids[x]:len(sense_ids)-x for x in range(len(sense_ids))}
    return results

### token overlap baseline
def tok_overlap_ranking(sent,definition_df):
    definition_df["tok_overlap"] = definition_df.apply (lambda row: token_overlap(sent,row["nlp_definition"]), axis=1)
    results = definition_df.set_index('sense_id').to_dict()["tok_overlap"]
    return results

def token_overlap(sent1,sent2):
    sent1 = set([tok.lemma_ for tok in sent1 if not tok.is_punct and not tok.is_stop])
    sent2 = set([tok.lemma_ for tok in sent2 if not tok.is_punct and not tok.is_stop])
    score = len(sent1 & sent2)
    return score

# sentence embedding similarity
def sent_embedding(sent,definition_df):
    definition_df["sent_embedding"] = definition_df.apply (lambda row: sent.similarity(row["nlp_definition"]), axis=1)
    results = definition_df.set_index('sense_id').to_dict()["sent_embedding"]
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
    results = definition_df.set_index('sense_id').to_dict()["w2v_lesk_ranking"]
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
    results = definition_df.set_index('sense_id').to_dict()["bert_lesk_ranking"]
    return results