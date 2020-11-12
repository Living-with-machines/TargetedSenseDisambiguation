import spacy

### preprocessing (for the moment only this) we need lg for having embeddings is spacy
nlp = spacy.load("en_core_web_lg", disable=['parser', 'ner'])

def preprocess(text):
    processed_text = nlp(text)
    return processed_text