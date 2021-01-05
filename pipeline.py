from pathlib import Path,PosixPath
import pandas as pd
from utils.dataset_download import *

def create_dataframe(lemma:str,
                    pos:str,
                    download_all:bool=False,
                    save_path:PosixPath=Path('./data')):
    """function that creates the required quotations dataframe
    it first harvet all senses related to given lemma id, than 
    obtains all quotations. Please note that the specified time 
    period only affects the retrieval the senses (i.e., we only 
    get senses who have an _overlap_ with the period bounded by start
    and end). Filtering of quotations is handled later by the
    obtain_quotations_for_senses function.

    Arguments:
        lemma (str): target lemma from which we expand 
        pos (str): restrict lemma to a specific part of speech
        download_ll (bool): flag used for testing the pipeline
                        if set to False, we only obtain the first
                        ten quotations (avoids overusing the OED API)
        save_path (PosixPath): save dataframes in the save_path folder
    """

    with open('./oed_experiments/oed_credentials.json') as f:
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

if __name__=="__main__":
    lemma, pos,download_all = parse_input_commands()
    print(lemma,pos, download_all)
    create_dataframe(lemma, pos,download_all)