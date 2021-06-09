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
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

WEMB_MODEL_PATH = "models/w2v_004/w2v_words.model"

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

        #df_test[f"bert_ts_nearest_centroid_sense_{vector_col}"] = df_test.apply(wsd.bert_ts_sense_centroid_vector,  
        #                senseid2label= senseid2label,
        #                ts_method='nearest',
        #                vector_col=vector_col,
        #                df_train = df_train, axis=1)

        df_test[f"bert_ts_weighted_centroid_sense_{vector_col}"] = df_test.apply(wsd.bert_ts_sense_centroid_vector,  
                        senseid2label= senseid2label,
                        ts_method='weighted',
                        vector_col=vector_col,
                        df_train = df_train, axis=1)

        # TO DO: uncomment this after merging with dev
        #df_test[f"bert_ts_weighted_centroid_sense_{vector_col}"] = df_test.apply(wsd.bert_ts_sense_centroid_vector,  
        #                senseid2label= senseid2label,
        #                ts_method='weighted_past',
        #                vector_col=vector_col,
        #                df_train = df_train, axis=1)

        print(f'[LOG] traing classifier for {senses} [BERT model = {vector_col}]' )
        X,y = list(df_train[vector_col].values), list(df_train.label.values)

        #svm_model = LinearSVC(random_state=0, C=.1, tol=1e-5,class_weight='balanced')
        #svm_model.fit(X,y)
        #df_test[f"bert_svm_{vector_col}"] = wsd.clf_svm(vector_col,df_test, svm_model)

        #perc_model = Perceptron(validation_fraction=.2, early_stopping=True,class_weight='balanced')
        #perc_model.fit(X,y)
        #df_test[f"bert_perceptron_{vector_col}"] = wsd.clf_perceptron(vector_col,df_test, perc_model)

        mlperc_model = MLPClassifier(validation_fraction=.2, early_stopping=True, solver='lbfgs',activation='relu')
        mlperc_model.fit(X,y)
        df_test[f"bert_ml_perceptron_{vector_col}"]  = wsd.clf_perceptron(vector_col,df_test, mlperc_model)


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
        results_filename,
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
    results_filename = results_filename + "+".join(sorted(relations)) + ".csv"
    Path(results_path).mkdir(parents=True, exist_ok=True)

    # IF df_test is None, create an empty DataFrame
    if not isinstance(df_test, type(None)):
        baselines = ["id_x",'label','year','quotation_id'] #'random','def_tok_overlap_ranking', 'sent_embedding', 'w2v_lesk_ranking'] # ,'svm_wemb_baseline'
        
        bert_methods = [[f"bert_centroid_sense_{vector_col}",
                        #f"bert_ts_nearest_centroid_sense_{vector_col}",
                        f"bert_ts_weighted_centroid_sense_{vector_col}",
                        f"bert_ml_perceptron_{vector_col}"
                            ] 
                                    for vector_col in  vector_cols]
        bert_methods = [i for tm in bert_methods for i in tm]

        out_df = df_test.filter(baselines + bert_methods, axis=1)
    else:
        out_df = pd.DataFrame()

    out_df.to_csv(os.path.join(results_path, results_filename), index=False)  

def run_experiment(direction='vertical'):
    if direction == 'vertical':
        RELATIONS = ['seed'] # 'synonym'
        EVAL_MODE = 'lemma' #'lemma_etal'
        RESULTS_PATH_BASE = 'results_curated_1920_seed'
    elif direction == 'horizontal':
        RELATIONS = ['seed','synonym'] # ''
        EVAL_MODE = 'lemma_etal' #'lemma_etal'
        RESULTS_PATH_BASE = 'results_curated_1920_syn'
    
    WEMB_MODEL = Word2Vec.load(WEMB_MODEL_PATH)
    TRAIN_ON_DEV = True

    # argument the change by experiment change
    FILTER_VAL = True
    FILTER_TEST = True

    VECTOR_COLS = ['vector_bert_base_-1,-2,-3,-4_mean',
                "vector_blert_base_-1,-2,-3,-4_mean",
                'vector_bert_1850_-1,-2,-3,-4_mean'
                ]

    START = 1760
    END = 1920
    
    FILTER_VAL = False
    FILTER_TEST = True

    #RESULTS_PATH_BASE = 'results_curated_1920_seed'

    words = {
    ("slave_sense_1_ethnicity",'slave','NN'): {"slave_nn01-22495496", "slave_nn01-22495604","slave_nn02-22498415"},
    ("slave_sense_2_humansubmission",'slave','NN'): {"slave_nn01-22495881","slave_nn01-22496175","slave_nn01-22496245","slave_nn01-22496365","slave_nn01-22498301"},
    ("slave_sense_3_object",'slave','NN'): {"slave_nn01-22496432","slave_nn01-22496462"},
    ("power_sense_1_ability",'power','NN'): {"power_nn01-28685348","power_nn01-28684730","power_nn01-28684973","power_nn01-28685458",
        "power_nn01-28685148","power_nn01-28686226","power_nn01-28684355","power_nn01-28685727","power_nn01-110297806","power_nn01-28684538"},
    ("power_sense_2_legal",'power','NN'): {"power_nn01-28685686","power_nn01-28685253","power_nn01-28688384","power_nn01-28687020",
        "power_nn01-28684910","power_nn01-110297806","power_nn01-28686314","power_nn01-28686446","power_nn01-28686005","power_nn01-28686566"},
    ("power_sense_3_technical","power","NN"): {"power_nn01-28688328","power_nn01-52944172","power_nn01-28685989","power_nn01-224965906",
        "power_nn01-28687020","power_nn01-28687898","power_nn01-52944158","power_nn01-28688218","power_nn01-28688482"},
    ("power_sense_4_maths","power","NN"): {"power_nn01-28687845","power_nn01-28687774","power_nn01-28687471","power_nn01-28687446",
        "power_nn01-28687561","power_nn01-28687732"},
    ("labour_sense_1_physical","labour","NN"): {"labour_nn01-39839017","labour_nn01-39838984","labour_nn01-39839322","labour_nn01-39838736",
        "labour_nn01-185646201","labour_nn01-185646093","labour_nn01-39839548","labour_nn01-39839598"},
    ("labour_sense_2_human","labour","NN"): {"labour_nn01-184772185","labour_nn01-184772185","labour_nn01-39839092"},
    ("labour_sense_3_figurative","labour","NN"): {"labour_nn01-39839738","labour_nn01-39839676"},
    ("labour_sense_4_political",'labour',"NN"): {"labour_nn01-39839157","labour_nn01-39839281"},
    ("machine_non_figurative","machine","NN"): {"machine_nn01-38473945","machine_nn01-38474233","machine_nn01-38474301","machine_nn01-38474548",
        "machine_nn01-38475099","machine_nn01-38475046","machine_nn01-38475013","machine_nn01-38474974","machine_nn01-38474877",
        "machine_nn01-38475494","machine_nn01-38474820","machine_nn01-38475164","machine_nn01-38475923","machine_nn01-38475286","machine_nn01-38474607"},
    ("machine_figurative","machine","NN"): {"machine_nn01-38474140","machine_nn01-38474405","machine_nn01-38475994","machine_nn01-38476096",
        "machine_nn01-38476316","machine_nn01-38476397","machine_nn01-38476566","machine_nn01-38476245","machine_nn01-38475835"}
}

    errors = []
    for name, senses in words.items():
        name,lemma,pos = name
        try:
            run(lemma, 
                pos, 
                senses, 
                start=START, 
                end=END,
                vector_cols=VECTOR_COLS,
                eval_mode=EVAL_MODE,
                relations=RELATIONS,
                train_on_dev=TRAIN_ON_DEV,
                wemb_model=WEMB_MODEL,
                filter_val=FILTER_VAL,
                filter_test=FILTER_TEST,
                results_filename=name,
                results_path_base=RESULTS_PATH_BASE)
        except Exception as e:
            print(name,e)
            errors.append(name)
    print("Done.")
    print("Errors with the following senses:")
    print(errors)
if __name__=="__main__":
    run_experiment(direction='vertical')
    run_experiment(direction='horizontal')
