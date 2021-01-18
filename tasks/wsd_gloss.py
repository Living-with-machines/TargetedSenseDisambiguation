import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path, PosixPath
from utils.classificaton_utils import binarize
from torch.optim.adam import Adam
from flair.datasets import CSVClassificationCorpus
from flair.data import MultiCorpus
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

# ---------------------------------------
# glossbert method ----------------------

def enclose_keyword(row:pd.Series,
                    enclose_token:str='"'):
    """enclose keyword with specific token to point
    learner towards to word it has to focus on
    """
    sentence = ''
    for i,c in enumerate(row.full_text):
        if i == int(row.keyword_offset):
            sentence+=enclose_token + ' '
        elif i ==int(row.keyword_offset + len(row.keyword)):
            sentence+= ' ' + enclose_token
        sentence+=c
    return sentence

def to_glossbert_format(df:pd.DataFrame):
    """convert rows in dataframe to GlossBERT format
    """

    def gloss_string(row:pd.Series, definition:str):
        """combine gloss with quoations and keyword
        """

        out_string=''
        if row.enclosed_quotation:
            out_string+=row.enclosed_quotation
        out_string+=' [SEP] '  
        out_string+=row.keyword+': '
        #if row.definition:
        out_string+=definition
        return out_string

    df['enclosed_quotation'] = df.apply(enclose_keyword, axis=1)
    
    rows = [] 
    for _ ,row in df.iterrows():
        rows.append([gloss_string(row, row.definition), "Yes", row.sense_id])
        definitions = df[df.lemma==row.lemma].definition.unique()
        for d in definitions:
            if d != row.definition:
                rows.append([gloss_string(row,d), "No",row.sense_id])
    
    return pd.DataFrame(rows, columns=['text','label','sense_id'])


def create_glossbert_data(lemma:str,
                        pos:str):
    """create glossbert data from quotations dataframe
    """

    df_quotations = pd.read_pickle(f'./data/sfrel_quotations_{lemma}_{pos}.pickle')
    df_quotations = df_quotations[~df_quotations.keyword_offset.isnull()]
    df_quotations = df_quotations[~df_quotations.definition.isnull()]#.reset_index(drop=True)
    df_glossbert = to_glossbert_format(df_quotations).sample(frac=1.0).reset_index(drop=True)
    print(df_glossbert.shape)
    # not sure if this is correct probably should split by positive example sentence?
    df_train, df_test = train_test_split(df_glossbert, test_size=0.2, random_state=42,shuffle=True) # , stratify=df_glossbert[['label']]
    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42,shuffle=True)
    
    train_data_path = Path("./data/training_data")
    train_data_path.mkdir(exist_ok=True)
    df_out_path = train_data_path / f'{lemma}_{pos}'
    df_out_path.mkdir(exist_ok=True)

    df_train.to_csv(df_out_path / 'train.csv', index = False, sep='\t')  
    df_val.to_csv(df_out_path / 'dev.csv', index = False, sep='\t')  
    df_test.to_csv(df_out_path / 'test.csv', index = False, sep='\t')

    return df_out_path

def train_glossbert(data_folder:PosixPath,
                    downsample:bool=False):

    column_name_map = {0: "text", 1: "label"}

    corpus = CSVClassificationCorpus(data_folder,
                                        column_name_map,
                                        skip_header=True,
                                        delimiter='\t',    # tab-separated files
                                        )
    
    if downsample:
        print('Downsampling.')
        corpus = corpus.downsample(0.1)
    
    label_dict = corpus.make_label_dictionary()
    print(label_dict)

    document_embeddings = TransformerDocumentEmbeddings('bert-base-uncased', fine_tune=True)

    classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, loss_weights={b"Yes":10, b"No":1}) # loss_weights={"1":10, "0":1}

    trainer = ModelTrainer(classifier, corpus, optimizer=Adam)

    trainer.train('models/classifier/glossbert',
            learning_rate=1e-3, # use very small learning rate
            mini_batch_size=16,
            embeddings_storage_mode='gpu',
            mini_batch_chunk_size=4, # optionally set this if transformer is too much for your machine
            max_epochs=50, # terminate after 5 epochs
            )


# ---------------------------------------
# multidataset training -----------------


def context_gloss_dfs(df:pd.DataFrame):
    """convert rows in dataframe to GlossBERT format
    """
    df = df[~df.keyword_offset.isnull()]
    df = df[~df.definition.isnull()].reset_index(drop=True)
    df['enclosed_quotation'] = df.apply(enclose_keyword, axis=1)
    df_gl =  df[['enclosed_quotation','definition','label']]
    return df_gl[['enclosed_quotation','label']],df_gl[['definition','label']].drop_duplicates()

def create_md_training_data(lemma:str, 
                            pos:str, 
                            senses:set, 
                            relations:list, 
                            experiment_id:int=0,
                            eval_mode:str='lemma_etal'):
    """create data for multidataset training
    """
    df_train, df_val, df_test = binarize(lemma,
                        pos,
                        senses, 
                        relations,
                        strict_filter=True,
                        start=1700,
                        end=2000,
                        eval_mode=eval_mode)

    data = list(map(context_gloss_dfs,[df_train, df_val, df_test]))

    train_data_path = Path("./data/training_data_md")
    train_data_path.mkdir(exist_ok=True)

    for context, gloss in data:
        for n, df in [('context',context),('gloss',gloss)]:

            df_out_path= train_data_path / f'{lemma}_{pos}_{experiment_id}_{n}'
            df_out_path.mkdir(exist_ok=True)

            df_train, df_test = train_test_split(df, 
                                        test_size=0.2, 
                                        random_state=42,
                                        shuffle=True,
                                        stratify=df[['label']]
                                        ) # 1st
                                                
            df_train, df_val = train_test_split(df_train, 
                                        test_size=0.1, 
                                        random_state=42,
                                        shuffle=True,
                                        stratify=df_train[['label']] # bug here, try to do the stratification better
                                        ) # 2nd
    
            df_train.to_csv(df_out_path / 'train.csv', index = False, sep='\t')  
            df_val.to_csv(df_out_path / 'dev.csv', index = False, sep='\t')  
            df_test.to_csv(df_out_path / 'test.csv', index = False, sep='\t')

def train_gloss_and_context(lemma:str,
                            pos:str,
                            experiment_id:int=0,
                            data_folder:PosixPath=Path("./data/training_data_md"),
                            downsample:bool=False):
    column_name_map = {0: "text", 1: "label"}

    context_corpus = CSVClassificationCorpus(data_folder / f"{lemma}_{pos}_{experiment_id}_context",
                                        column_name_map,
                                        skip_header=True,
                                        delimiter='\t',    # tab-separated files
                                        )
    gloss_corpus = CSVClassificationCorpus(data_folder / f"{lemma}_{pos}_{experiment_id}_gloss",
                                        column_name_map,
                                        skip_header=True,
                                        delimiter='\t',    # tab-separated files
                                        )
    
    corpus = MultiCorpus([context_corpus, gloss_corpus])
    
    if downsample:
        print('Downsampling...')
        corpus = corpus.downsample(0.1)
    
    label_dict = corpus.make_label_dictionary()
    print(label_dict)

    document_embeddings = TransformerDocumentEmbeddings('bert-base-uncased', fine_tune=True)

    classifier = TextClassifier(document_embeddings, label_dictionary=label_dict) # loss_weights={"1":10, "0":1}

    trainer = ModelTrainer(classifier, corpus, optimizer=Adam)

    trainer.train('models/classifier/glossbert',
            learning_rate=1e-3, # use very small learning rate
            mini_batch_size=16,
            embeddings_storage_mode='gpu',
            mini_batch_chunk_size=4, # optionally set this if transformer is too much for your machine
            max_epochs=50, # terminate after 5 epochs
            )