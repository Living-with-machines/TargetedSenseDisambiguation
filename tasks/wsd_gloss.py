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
                    enclose_token:str='"') -> str:
    """enclose keyword with specific token to point
    learner towards to word it has to focus on. this 
    is part of the weak supervision when learning
    from context/quotations.

    Arguments:
        row (pd.Series): row of quotations dataframe
        enclose_token (str): use token to mark target expression
                    effectively this serves begin and end token

    Returns:
        quotation with target token marked by `enclose_token`
    """
    sentence = ''
    for i,c in enumerate(row.full_text):
        if i == int(row.keyword_offset):
            sentence+=enclose_token + ' '
        elif i ==int(row.keyword_offset + len(row.keyword)):
            sentence+= ' ' + enclose_token
        sentence+=c
    return sentence

def to_glossbert_format(df:pd.DataFrame) -> pd.DataFrame:
    """convert rows in dataframe to GlossBERT format
    Argument:
        df (pd.DataFrame): quotations dataframe

    Returns:
        pd.DataFrame with format confirming the 
        GlossBERT template
    """

    def gloss_string(row:pd.Series, definition:str) -> str:
        """combine gloss with quotations and keyword
        
        Arguments:
            row (pd.Series): row of dataframe
            definition (str): definition to use as gloss
        
        Returns:
            out_string that combines as quotation/context
            with a gloss seperated by [SEP]
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

    # create labelled observations 1 of the context matches the definition
    # 0 for the other cases (this method used weak supervision)
    for _ ,row in df.iterrows():
        rows.append([gloss_string(row, row.definition), "1", row.sense_id])
        definitions = df[df.lemma==row.lemma].definition.unique()
        for d in definitions:
            if d != row.definition:
                rows.append([gloss_string(row,d), "0",row.sense_id])
    
    return pd.DataFrame(rows, columns=['text','label','sense_id'])


def create_glossbert_data(lemma:str,
                        pos:str) -> PosixPath:
    """Create glossbert data from quotations dataframe
    Arguments:
        lemma (str): lemma 
        pos (str): part-of-speech

    Return:
        path as PosixPath to location where data is stored
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
                    downsample:bool=False) -> bool:
    """train as GlossBERT model
    Arguments:
        data_folder (PosixPath): folder where train/dev and 
                    test set are stored as csv files
        downsample (bool): if True we use only ten per cent
                        of the data for training and testing
                        primarily used for demo puroposes
                
    Return:
        return True after training
    """

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
    
    return True


# ---------------------------------------
# multidataset training -----------------


def context_gloss_dfs(df:pd.DataFrame) -> tuple:
    """split the quotations dataframe in a context/quotation 
    and a gloss dataframe.

    Arguments:
        df (pd.DataFrame): quotations dataframe
    Returns:
        a tuple in the format (context_df, gloss_df)
    """
    df = df[~df.keyword_offset.isnull()]
    df = df[~df.definition.isnull()].reset_index(drop=True)
    df['enclosed_quotation'] = df.apply(enclose_keyword, axis=1)
    df_gl =  df[['enclosed_quotation','definition','label']]
    return (df_gl[['enclosed_quotation','label']],
            df_gl[['definition','label']].drop_duplicates())

def create_md_training_data(lemma:str, 
                            pos:str, 
                            senses:set, 
                            relations:list, 
                            experiment_id:int=0,
                            eval_mode:str='lemma_etal'):
    """create data for multidataset training in which
    we train a model simultaneously on quotations and glosses.

    Arguments:
        lemma (str): lemma
        pos (str): part-of-speech
        senses (set): senses that define the positive class
        relations (list): relation used for expanding the senses
        experiment_id (int): integer identifier used as id
        eval_mode (str): evalation mode (lemma or lemma_etal)

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
                                        stratify=df_train[['label']] 
                                        ) # 2nd
    
            df_train.to_csv(df_out_path / 'train.csv', index = False, sep='\t')  
            df_val.to_csv(df_out_path / 'dev.csv', index = False, sep='\t')  
            df_test.to_csv(df_out_path / 'test.csv', index = False, sep='\t')

def train_gloss_and_context(lemma:str,
                            pos:str,
                            experiment_id:int=0,
                            data_folder:PosixPath=Path("./data/training_data_md"),
                            downsample:bool=False) -> bool:
    """fine-tune a transformer model on both the context and the gloss

    Arguments:
        lemma (str): lemma
        pos (str): part-of-speech
        experiment_id (int): integer used to identify experiment
        data_folder (PosixPath): main folder for storing the 
                context and gloss folder
        downsample (bool): if True we use only 10% of the data
                for training and testing

    Returns:
        returns True after model has finished training
    """

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
    return True