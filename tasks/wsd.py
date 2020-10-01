from random import shuffle

def random_predict(definition_df):
    sense_ids = definition_df["sense_id"].tolist()
    shuffle(sense_ids)
    p = sense_ids[0]
    return p