import spacy
from random import shuffle

nlp = spacy.load("en_core_web_sm")

def random_predict(definition_df):
    sense_ids = definition_df["sense_id"].tolist()
    shuffle(sense_ids)
    p = sense_ids[0]
    return p

def tok_overlap_ranking(sent,definition_df):
    definition_df["tok_overlap"] = definition_df.apply (lambda row: token_overlap(sent,row["definition"]), axis=1)
    p = definition_df.iloc[definition_df['tok_overlap'].idxmax()]["sense_id"]
    return p

def token_overlap(sent1,sent2):
    # we could do this text processing before as a preproc step
    sent1 = set([tok.text.lower() for tok in nlp(sent1)])
    sent2 = set([tok.text.lower() for tok in nlp(sent2)])
    score = len(sent1 & sent2)
    return score
