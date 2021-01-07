import spacy
import numpy as np

### preprocessing (for the moment only this) we need lg for having embeddings is spacy
nlp = spacy.load("en_core_web_lg", disable=['parser', 'ner'])

def preprocess(text):
    processed_text = nlp(text)
    return processed_text

def avg_embedding(text,emb_model):
    text = [tok.lemma_ for tok in text if not tok.is_punct and not tok.is_stop]

    doc_embed = []
    for word in text:
            try:
                embed_word = emb_model[word]
                doc_embed.append(embed_word)
            except KeyError:
                continue   
    if len(doc_embed)>0:
        avg = [float(sum(col))/len(col) for col in zip(*doc_embed)]
        avg = np.array(avg)
        return avg
    else:
        return np.zeros(emb_model.vector_size)