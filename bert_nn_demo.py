from utils.classificaton_utils import *
from pathlib import Path
from flair.embeddings import TransformerWordEmbeddings

if __name__=="__main__":
    embedding_type = TransformerWordEmbeddings('bert-base-uncased',
                                        layers='-1,-2,-3,-4', # '-1,-2,-3,-4'
                                        pooling_operation='mean')
    
    path = Path('./data/quotations_all_machine_nn01.pickle')

    quotations = prepare_data(path,embedding_type)
    
    quotation_machine = quotations[quotations.word_id=="machine_nn01"]
    print(quotation_machine.shape)
    query_vector = quotation_machine.iloc[0].vector
    print(bert_avg_quot_nn_wsd(query_vector, quotation_machine))	