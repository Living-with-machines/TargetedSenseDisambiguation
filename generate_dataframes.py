from pathlib import Path,PosixPath
import pandas as pd
from utils.dataset_download import *
from utils.classificaton_utils import vectorize_target_expressions


BERT_1850 = './models/FT_bert_base_uncased_before_1850_v001'
BERT_1900 = './models/FT_bert_base_uncased_all_books_v002'
CREDENTIALS_FILE = './data/oed_credentials.json'

def create_dataframe(lemma:str,
                    pos:str,
                    download_all:bool=False,
                    save_path:PosixPath=Path('./data')):
    """function that creates the required quotations dataframe
    it first harvet all senses related to given lemma id, than 
    obtains all quotations. Filtering of quotations is handled later by the
    obtain_quotations_for_senses function.

    Example usage
        python pipeline.py --lemma='democracy' --pos='NN'

    Arguments:
        lemma (str): target lemma from which we expand 
        pos (str): restrict lemma to a specific part of speech
        download_ll (bool): flag used for testing the pipeline
                        if set to False, we only obtain the first
                        ten quotations (avoids overusing the OED API)
        save_path (PosixPath): save dataframes in the save_path folder
    """

    with open(CREDENTIALS_FILE) as f:
        auth = json.load(f)

    # create folder if it doesn't yet exist
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)
    
    # obtain all related senses
    extend_from_lemma(auth,lemma, pos)
    
    # get all quotations for words observed when
    # retrieve the lemmas, this is an incluse harvesting
    # to obtain also senses that shave the same surface form and POS
    # but are not related to machines
    harvest_data_from_extended_senses(auth,f"{lemma}_{pos}", download_all=download_all)

    quotations_path = f"./data/sfrel_quotations_{lemma}_{pos}.pickle"

    # [WARNING] this script requires you to add path the BERT_1850 and BERT_1900 models!
    embedding_methods = {'bert_base': {"path":'bert-base-uncased',
                                'layers':'-1,-2,-3,-4',
                                'pooling_operation':'mean'},
                        'blert': {"path":BERT_1900, # !! specify path to BERT_1900 model
                                'layers':'-1,-2,-3,-4',
                                'pooling_operation':'mean'},
                        'bert_1850':{"path":BERT_1850, # !! specify path to BERT_1850 model
                                'layers':'-1,-2,-3,-4',
                                'pooling_operation':'mean'}
                        }
    quotations = vectorize_target_expressions(quotations_path,embedding_methods)
if __name__=="__main__":
    lemmas = [('machine','NN')]#[('woman','NN'), ('man','NN'),('apple','NN'), ('anger','NN'), ('happiness','NN'),
            #('nation','NN'),('art','NN'), ('technology','NN'), ('labour','NN'),
            #('power','NN'), ('democracy','NN'), ('slave','NN')]
    
    for lemma, pos in lemmas:
        print(lemma,pos)
        create_dataframe(lemma, pos,download_all=True)