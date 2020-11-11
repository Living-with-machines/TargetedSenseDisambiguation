import spacy
from random import shuffle
from sklearn.metrics import precision_recall_fscore_support

nlp = spacy.load("en_core_web_sm")

def eval(ranking,gold):
    preds = []
    for line in ranking:
        sort_ranking = [[k, v] for k, v in sorted(line.items(), key=lambda item: item[1],reverse=True)]
        p = sort_ranking[0][0]
        preds.append(p)
    p,r,f1 = [round(x,3) for x in precision_recall_fscore_support(gold,preds, average='macro')[:3]]
    microf1 = round(precision_recall_fscore_support(gold,preds, average='micro')[2],3)
    return p,r,f1,microf1

def random_predict(definition_df):
    sense_ids = definition_df["sense_id"].tolist()
    shuffle(sense_ids)
    results = {sense_ids[x]:len(sense_ids)-x for x in range(len(sense_ids))}
    return results

def tok_overlap_ranking(sent,definition_df):
    definition_df["tok_overlap"] = definition_df.apply (lambda row: token_overlap(sent,row["definition"]), axis=1)
#    results = definition_df.iloc[definition_df['tok_overlap'].idxmax()]["sense_id"]
    results = definition_df.set_index('sense_id').to_dict()["tok_overlap"]
    return results

def token_overlap(sent1,sent2):
    # we could do this text processing before as a preproc step
    sent1 = set([tok.text.lower() for tok in nlp(sent1)])
    sent2 = set([tok.text.lower() for tok in nlp(sent2)])
    score = len(sent1 & sent2)
    return score
