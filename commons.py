import torch, pickle, os, tqdm
from nltk.translate.meteor_score import meteor_score
import numpy as np
import visdom
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
basefilename = None
VISDOM_PORT = 8080
VISDOM_SERVER = "192.168.0.109"
VISDOM = visdom.Visdom(server=VISDOM_SERVER, port=VISDOM_PORT)

def cuda(obj):
    if torch.cuda.is_available():
        obj = obj.cuda()
    return obj

def calculate_accuracy(scores, targets, k=1):
    f = torch.FloatTensor()
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()
    
    return (correct_total.item())/batch_size


def prepare_embeddings(embedding_path):
    vec_save_loc = "output/"+embedding_path.split("/")[-1]+"_W2VEC.pkl"
    
    if os.path.exists(vec_save_loc):
        with open(vec_save_loc, 'rb') as f:
            w2vec = pickle.load(f)
            shape = len(next(iter(w2vec.values())))
    else:
        w2vec = {}
        print("Loading word vectors...")
        with open(embedding_path) as f:
            lines = f.readlines()
            if embedding_path.endswith(".vec"):
                lines.pop(0)
            shape = len(lines[0].split(' '))-1
            for line in tqdm.tqdm(lines):
                items = line.strip().split(' ')
                token = items[0]
                vector = np.array(items[1:]).astype(float)
                w2vec[token] = vector
        with open(vec_save_loc, 'wb') as f:
            pickle.dump(w2vec, f)
    return w2vec, shape


def get_matched_embeddings(all_embeddings, word_map, dim):
    rev_word_map = {v:k for k,v in word_map.items()}
    embeddings = list()
    
    for i in range(len(rev_word_map)):
        word = rev_word_map[i]
        if word not in all_embeddings:
            embeddings.append(np.random.uniform(-1.2, 1.2, size=dim))
        else:
            embeddings.append(all_embeddings[word])
    
    return np.stack(embeddings)