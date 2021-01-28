import os
import sys
import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec
from utils.classificaton_utils import run_all

if __name__=="__main__":
    

    RELATIONS = ['seed','synonym']
    EVAL_MODE = 'lemma_etal'
    WEMB_MODEL = Word2Vec.load("models/w2v_004/w2v_words.model")
    TRAIN_ON_DEV = True

    # argument the change by experiment change

    VECTOR_COLS = ['vector_bert_base_-1,-2,-3,-4_mean',
                "vector_blert_base_-1,-2,-3,-4_mean",
                'vector_bert_1850_-1,-2,-3,-4_mean'
                ]

    START = 1760 
    END = 1850 
    
    RESULTS_PATH_BASE = "results_ts_1850"
    FILTER_VAL = False
    FILTER_TEST = True
    
    words = [['anger',"NN"],["apple","NN"],["art","NN"],["democracy","NN"],
            ["happiness","NN"],["labour","NN"],["machine","NN"],["man","NN"],
            ["nation","NN"],["power","NN"],["slave","NN"],['woman','NN']]

    errors = []
    
    for lemma, pos in words:
        quotations_path = f"./data/sfrel_quotations_{lemma}_{pos}.pickle"
        lemma_senses = pd.read_pickle(f'./data/lemma_senses_{lemma}_{pos}.pickle')
    
        # this is the index of the lemma id <-- we could remove this later
        #idx = "01"
        #senses = set(lemma_senses[lemma_senses.word_id==f'{lemma}_{pos.lower()}{idx}'].id)
        senses = set(lemma_senses[lemma_senses.word_id.str.startswith(f'{lemma}_{pos.lower()}')]["id"])
        
        for sense in tqdm(list(senses)):
        
            try:
                run_all(lemma, 
                    pos, 
                    {sense}, 
                    start=START, 
                    end=END,
                    vector_cols=VECTOR_COLS,
                    eval_mode=EVAL_MODE,
                    relations=RELATIONS,
                    train_on_dev=TRAIN_ON_DEV,
                    wemb_model=WEMB_MODEL,
                    filter_val=FILTER_VAL,
                    filter_test=FILTER_TEST,
                    results_path_base=RESULTS_PATH_BASE)
            except Exception as e:
                print(sense,e)
                errors.append(sense)
    print("Done.")
    print("Errors with the following senses:")
    print(errors)