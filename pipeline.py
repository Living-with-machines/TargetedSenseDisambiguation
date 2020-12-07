from argparse import ArgumentParser
from pathlib import Path,PosixPath
import pandas as pd
from utils.dataset_download import *

def create_dataframe(lemma_id:str,
                    start:int,
                    end:int,
                    download_all:bool=False,
                    save_path=Path('./data')):
    """function that creates the required quotations dataframe
    """

    with open('./oed_experiments/oed_credentials.json') as f:
        auth = json.load(f)

    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)
    
    extend_from_lemma(auth,lemma_id,start,end)
    
    harvest_quotations(auth,lemma_id,level='word', download_all=download_all)

if __name__=="__main__":
    lemma_id,start,end,download_all = parse_input_commands()
    print(lemma_id,start,end, download_all)
    create_dataframe(lemma_id,start,end,download_all)