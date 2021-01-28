import os
import sys
import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec
from utils.classificaton_utils import run_all

if __name__=="__main__":
    

    RELATIONS = ['seed'] # 'synonym'
    EVAL_MODE = 'lemma' #'lemma_etal'
    WEMB_MODEL = Word2Vec.load("models/w2v_004/w2v_words.model")
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

    RESULTS_PATH_BASE = 'results_curated_1920_seed'

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
    ("macchine_figurative","machine","NN"): {"machine_nn01-38474140","machine_nn01-38474405","machine_nn01-38475994","machine_nn01-38476096",
        "machine_nn01-38476316","machine_nn01-38476397","machine_nn01-38476566","machine_nn01-38476245","machine_nn01-38475835"}}

    errors = []
    for name, senses in words.items():
        name,lemma,pos = name
        try:
            run_all(lemma, 
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
                #results_filename=name,
                results_path_base=RESULTS_PATH_BASE,
                exp=2)
        except Exception as e:
            print(name,e)
            errors.append(name)
    print("Done.")
    print("Errors with the following senses:")
    print(errors)