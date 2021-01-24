import os
import pandas as pd
import numpy as np
from tasks import wsd
from pathlib import Path
from utils import nlp_tools
from parhugin import multiFunc
from gensim.models import Word2Vec
from utils.classificaton_utils import binarize,generate_definition_df, vectorize_target_expressions
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

def eval_sense(lemma,
                pos,
                senses,
                start,
                end,
                train_on_dev,
                eval_mode,
                relations,
                vector_cols,
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
                                    filter_test_by_year=filter_test,
                                    strict_filter=True)

    # no quotations for sense and timeframe
    if df_train is None:
        return None
    
    if train_on_dev:
        df_train = pd.concat([df_train, df_val], axis=0)
        df_train.reset_index(inplace=True,drop=True)

    df_train["nlp_full_text"] = df_train.apply (lambda row: nlp_tools.preprocess(row["full_text"]), axis=1)
    df_val["nlp_full_text"] = df_val.apply (lambda row: nlp_tools.preprocess(row["full_text"]), axis=1)
    df_test["nlp_full_text"] = df_test.apply (lambda row: nlp_tools.preprocess(row["full_text"]), axis=1)

    # random
    print(f'[LOG] computing baselines for {senses}')
    df_test["random"] = df_test.apply (lambda row: wsd.random_predict(), axis=1)

    # retrieve and process definitions            
    df_selected_senses = generate_definition_df(df_train,lemma,eval_mode=eval_mode)
    df_selected_senses["nlp_definition"] = df_selected_senses.apply (lambda row: nlp_tools.preprocess(row["definition"]), axis=1)

    # token overlap
    df_test["def_tok_overlap_ranking"] = df_test.apply (lambda row: wsd.tok_overlap_ranking(row["nlp_full_text"], df_selected_senses), axis=1)

    # spacy sentence embeddings
    df_test["sent_embedding"] = df_test.apply (lambda row: wsd.sent_embedding(row["nlp_full_text"], df_selected_senses), axis=1)

    #w2v lesk
    df_test["w2v_lesk_ranking"] = df_test.apply (lambda row: wsd.w2v_lesk_ranking(row["nlp_full_text"], df_selected_senses, wemb_model), axis=1)

    #Bert lesk      
    #df_test["bert_lesk_ranking"] = df_test.apply (lambda row: wsd.bert_lesk_ranking(row["text"]["full_text"], df_selected_senses, bert_sentsim_model), axis=1)

    # supervised baselined (w-emb SVM) - careful this is a 19thC BL model
    df_test["svm_wemb_baseline"] = wsd.svm_wemb_baseline(df_train,df_test,wemb_model)

    for vector_col in vector_cols:
        print(f'[LOG] computing centoids for {senses} [BERT model = {vector_col}]' )
        df_test[f"bert_binary_centroid_{vector_col}"] = df_test.apply(wsd.bert_binary_centroid_vector, 
                                        df_train = df_train, 
                                        vector_col=vector_col,
                                        return_ranking=False, axis=1)

        senseid2label = dict(df_test[['sense_id','label']].values)
        df_test[f"bert_centroid_sense_{vector_col}"] = df_test.apply(wsd.bert_sense_centroid_vector,  
                                                    senseid2label= senseid2label,
                                                    vector_col=vector_col,
                                                    df_train = df_train, axis=1)

        centroid_vectors = df_train.groupby('label')[vector_col].apply(np.mean,axis=0)
        sem_axis = centroid_vectors[1] - centroid_vectors[0] 
        df_test[f"bert_contrast_{vector_col}"] = df_test[vector_col].apply(wsd.bert_semaxis_vector,
                                                    sem_axis=sem_axis,
                                                    threshold=.0)

        df_test[f"bert_ts_binary_centroid_{vector_col}"] = df_test.apply(wsd.bert_ts_binary_centroid_vector, 
                                                        df_train=df_train, 
                                                        ts_method='nearest',
                                                        vector_col=vector_col,
                                                        axis=1)

        senseid2label = dict(df_test[['sense_id','label']].values)
        df_test[f"bert_ts_centroid_sense_{vector_col}"] = df_test.apply(wsd.bert_ts_sense_centroid_vector,  
                        senseid2label= senseid2label,
                        ts_method='nearest',
                        vector_col=vector_col,
                        df_train = df_train, axis=1)

        print(f'[LOG] traing classifier for {senses} [BERT model = {vector_col}]' )
        X,y = list(df_train[vector_col].values), list(df_train.label.values)

        svm_model = LinearSVC(random_state=0, C=.1, tol=1e-5,class_weight='balanced')
        svm_model.fit(X,y)
        df_test[f"bert_svm_{vector_col}"] = wsd.clf_svm(vector_col,df_test, svm_model)

        perc_model = Perceptron(validation_fraction=.2, early_stopping=True,class_weight='balanced')
        perc_model.fit(X,y)
        df_test[f"bert_perceptron_{vector_col}"] = wsd.clf_perceptron(vector_col,df_test, perc_model)

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
                filter_test=filter_test,
                wemb_model=wemb_model)

    results_path = os.path.join(results_path_base, f"{lemma}_{pos}", eval_mode)
    results_filename = '_'.join(senses) + "~" + "+".join(sorted(relations)) + ".csv"
    Path(results_path).mkdir(parents=True, exist_ok=True)

    # IF df_test is None, create an empty DataFrame
    if not isinstance(df_test, type(None)):
        baselines = ['id_x','label','random','def_tok_overlap_ranking', 'sent_embedding', 'w2v_lesk_ranking',                        'svm_wemb_baseline']
        bert_methods = [[f"bert_binary_centroid_{vector_col}",f"bert_centroid_sense_{vector_col}",f"bert_contrast_{vector_col}",
                        f"bert_ts_binary_centroid_{vector_col}",f"bert_ts_centroid_sense_{vector_col}",
                        f"bert_svm_{vector_col}",f"bert_perceptron_{vector_col}",f"bert_ml_perceptron_{vector_col}"
                        ] 
                                    for vector_col in  vector_cols]
        bert_methods = [i for tm in bert_methods for i in tm]

        out_df = df_test.filter(baselines + bert_methods, axis=1)
    else:
        out_df = pd.DataFrame()

    out_df.to_csv(os.path.join(results_path, results_filename), index=False)  

if __name__=="__main__":
    # arguments that remain constant for all experiments
    VECTOR_COLS = ['vector_bert_base_-1,-2,-3,-4_mean',
                "vector_blert_base_-1,-2,-3,-4_mean",
                'vector_bert_1850_-1,-2,-3,-4_mean']
    RELATIONS = ['seed','synonym']
    EVAL_MODE = 'lemma_etal'
    WEMB_MODEL = Word2Vec.load("models/w2v_004/w2v_words.model")

    # argument may change
    TRAIN_ON_DEV = True
    FILTER_TEST = True

    # arguments that vary by experiment
    START = 1760
    END = 1920 # 1850 = results_2 1920 = results 2000 = results_3

    RESULTS_PATH_BASE = "results_2"

    # arguments that vary for each run
    #words = [['anger',"NN"],["apple","NN"],["art","NN"],["democracy","NN"],
    #        ["happiness","NN"],["labour","NN"],["machine","NN"],["man","NN"],
    #        ["nation","NN"],["power","NN"],["slave","NN"],['woman','NN']]

    words = [["democracy","NN"],["labour","NN"],["machine","NN"],
            ["nation","NN"],["power","NN"],["slave","NN"],['woman','NN']]
    errors = []

    for lemma, pos in words:
        quotations_path = f"./data/sfrel_quotations_{lemma}_{pos}.pickle"
        lemma_senses = pd.read_pickle(f'./data/lemma_senses_{lemma}_{pos}.pickle')
    
        # this is the index of the lemma id <-- we could remove this later
        idx = "01"
        senses = set(lemma_senses[lemma_senses.word_id==f'{lemma}_{pos.lower()}{idx}'].id)
        
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
                    filter_test=FILTER_TEST,
                    results_path_base=RESULTS_PATH_BASE)
            except Exception as e:
                print(sense,e)
                errors.append(sense)
    print("Done.")
    print("Errors with the following senses:")
    print(errors)