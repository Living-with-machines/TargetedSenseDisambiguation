from utils.classificaton_utils import *
from pathlib import Path
from flair.embeddings import TransformerWordEmbeddings

def test_bert_001(model_name: str='bert-base-uncased',
                layers: str="-1",
                pooling_operation: str="mean",
                dataframe_file: str="./data/quotations_all_machine_nn01.pickle",
                word_id: str="machine_nn01"):

    print(f"[INFO] Prepare data using {model_name}.")
    embedding_type = TransformerWordEmbeddings(model_name,
                                            layers=layers,
                                            pooling_operation=pooling_operation)

    # path should point to a file with all quotations
    # this file was created by the harvest_quotations function
    path = Path(dataframe_file)

    quotations = prepare_data(path, embedding_type)

    print(f"[INFO] Extract quotations for word_id: {word_id}")    
    quotation_target = quotations[quotations.word_id==word_id]
    print(f"Shape: {quotation_target.shape}")

    query_vector = quotation_target.iloc[0].vector
    print(bert_avg_quot_nn_wsd(query_vector, quotation_target))	

if __name__=="__main__":
    test_bert_001()
