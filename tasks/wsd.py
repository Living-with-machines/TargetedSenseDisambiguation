import spacy
from random import shuffle
from sklearn.metrics import precision_recall_fscore_support


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


### preprocessing (for the moment only this)
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

def preprocess(text):
    processed_text = nlp(text)
    return processed_text


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
    sent1 = set([tok.text.lower() for tok in sent1])
    sent2 = set([tok.text.lower() for tok in sent2])
    score = len(sent1 & sent2)
    return score



