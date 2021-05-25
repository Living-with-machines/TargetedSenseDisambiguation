import os
import pandas as pd
import numpy as np
from tasks import wsd
from pathlib import Path
#from utils import nlp_tools
#from parhugin import multiFunc
from gensim.models import Word2Vec
from utils.classificaton_utils import binarize#,generate_definition_df, vectorize_target_expressions
from tqdm import tqdm
#from sklearn.svm import LinearSVC
#from sklearn.linear_model import Perceptron
#from sklearn.neural_network import MLPClassifier

def eval_sense(lemma,
                pos,
                senses,
                start,
                end,
                train_on_dev,
                eval_mode,
                relations,
                vector_cols,
                filter_val,
                filter_test,
                wemb_model):

    print(f'STARTING AT {start}; ENDING AT {end}')

    df_train, df_val, df_test = binarize(lemma=lemma,
                                    pos=pos,
                                    senses=senses, 
                                    start=start,
                                    end=end,
                                    relations=relations,
                                    eval_mode=eval_mode,
                                    filter_val_by_year=filter_val,
                                    filter_test_by_year=filter_test,
                                    strict_filter=True)

    # no quotations for sense and timeframe
    if df_train is None:
        return None
    
    if train_on_dev:
        df_train = pd.concat([df_train, df_val], axis=0)
        df_train.reset_index(inplace=True,drop=True)

    

    for vector_col in vector_cols:
        
        senseid2label = dict(df_test[['sense_id','label']].values)
        df_test[f"bert_centroid_sense_{vector_col}"] = df_test.apply(wsd.bert_sense_centroid_vector,  
                                                    senseid2label= senseid2label,
                                                    vector_col=vector_col,
                                                    df_train = df_train, axis=1)

        df_test[f"bert_ts_nearest_centroid_sense_{vector_col}"] = df_test.apply(wsd.bert_ts_sense_centroid_vector,  
                        senseid2label= senseid2label,
                        ts_method='nearest',
                        vector_col=vector_col,
                        df_train = df_train, axis=1)

        df_test[f"bert_ts_weighted_centroid_sense_{vector_col}"] = df_test.apply(wsd.bert_ts_sense_centroid_vector,  
                        senseid2label= senseid2label,
                        ts_method='weighted',
                        vector_col=vector_col,
                        df_train = df_train, axis=1)

        # Inspired by https://recordlinkage.readthedocs.io/en/latest/ref-compare.html#recordlinkage.compare.Numeric
        df_test[f"bert_ts_weighted_centroid_sense_gauss_{vector_col}"] = df_test.apply(wsd.bert_ts_sense_centroid_vector,  
                        senseid2label= senseid2label,
                        ts_method='weighted_gauss',
                        vector_col=vector_col,
                        df_train = df_train, axis=1)

        # TO DO: uncomment this after merging with dev
        df_test[f"bert_ts_weighted_centroid_sense_past_{vector_col}"] = df_test.apply(wsd.bert_ts_sense_centroid_vector,  
                       senseid2label= senseid2label,
                       ts_method='weighted_past',
                       vector_col=vector_col,
                       df_train = df_train, axis=1)

        


    return df_test

def run(lemma, 
        pos, 
        senses, 
        start, 
        end,
        vector_cols,
        eval_mode,
        relations,
        train_on_dev,
        wemb_model,
        filter_val,
        filter_test,
        results_path_base):
        
    df_test = eval_sense(lemma=lemma,
                pos=pos,
                senses=senses,
                start=start,
                end=end,
                train_on_dev=train_on_dev,
                eval_mode=eval_mode,
                relations=relations,
                vector_cols=vector_cols,
                filter_val=filter_val,
                filter_test=filter_test,
                wemb_model=wemb_model)

    results_path = os.path.join(results_path_base, f"{lemma}_{pos}", eval_mode)
    results_filename = '_'.join(senses) + "~" + "+".join(sorted(relations)) + ".csv"
    Path(results_path).mkdir(parents=True, exist_ok=True)

    # IF df_test is None, create an empty DataFrame
    if not isinstance(df_test, type(None)):
        baselines = ["id_x",'label','year','quotation_id']
        bert_methods = [[
                        f"bert_centroid_sense_{vector_col}",
                        f"bert_ts_nearest_centroid_sense_{vector_col}",
                        f"bert_ts_weighted_centroid_sense_{vector_col}",
                        f"bert_ts_weighted_centroid_sense_past_{vector_col}",
                        f"bert_ts_weighted_centroid_sense_gauss_{vector_col}",
                        ] 
                                    for vector_col in  vector_cols]
        bert_methods = [i for tm in bert_methods for i in tm]

        out_df = df_test.filter(baselines + bert_methods, axis=1)
    else:
        out_df = pd.DataFrame()

    out_df.to_csv(os.path.join(results_path, results_filename), index=False)  

def run_experiment(END):
    RELATIONS = ['seed','synonym']
    EVAL_MODE = 'lemma_etal'
    WEMB_MODEL = Word2Vec.load("models/word2vec/w2v_1760_1900/w2v_words.model")
    TRAIN_ON_DEV = True

    # argument the change by experiment change

    VECTOR_COLS = ['vector_bert_base_-1,-2,-3,-4_mean',
                "vector_blert_-1,-2,-3,-4_mean",
                'vector_bert_1850_-1,-2,-3,-4_mean'
                ]

    START = 1760

    
    #END = 1850 
    RESULTS_PATH_BASE = f"results_ts_{END}"
    FILTER_VAL = False
    FILTER_TEST = True
    
    words = [['anger',"NN"],["apple","NN"],["art","NN"],["democracy","NN"],
            ["happiness","NN"],["labour","NN"],["machine","NN"],["man","NN"],
            ["nation","NN"],["power","NN"],["slave","NN"],['woman','NN']]
    
    # words = [["machine","NN"]]

    errors = []
    
    for lemma, pos in words:

        print(lemma, pos)

        quotations_path = f"./data/sfrel_quotations_{lemma}_{pos}.pickle"
        lemma_senses = pd.read_pickle(f'./data/lemma_senses_{lemma}_{pos}.pickle')
    
        # this is the index of the lemma id <-- we could remove this later
        #idx = "01"
        #senses = set(lemma_senses[lemma_senses.word_id==f'{lemma}_{pos.lower()}{idx}'].id)
        senses = set(lemma_senses[lemma_senses.word_id.str.startswith(f'{lemma}_{pos.lower()}')]["id"])
        
        for sense in tqdm(list(senses)):
        
            try:
                run(lemma, 
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

if __name__=="__main__":
    run_experiment(1850)
    run_experiment(1920)
