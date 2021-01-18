from tasks.wsd_gloss import create_glossbert_data, train_glossbert
import sys

def run(lemma,pos):
    data_path = create_glossbert_data(lemma,pos)
    train_glossbert(data_path,downsample=True)

if __name__=="__main__":
    lemma,pos = sys.argv[1],sys.argv[2]
    run(lemma,pos)
