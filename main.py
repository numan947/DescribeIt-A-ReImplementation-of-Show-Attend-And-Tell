from train import train
from data import ImageCaptionDataset
from models import Encoder, Decoder
from torch.utils.data import DataLoader
import torch
import pandas as pd
import json, os
from eval import evaluate, generate_caption
from torchvision.transforms import transforms
from glob import glob

ALL_SEEDS = [43, 947, 94743]

SEED = 0
DATA_FOLDER = "./output/"
DATA_NAME = ["FLICKR8K", "FLICKR30K"]
CAPTIONS_PER_IMAGE = 5
MIN_WORD_FREQ = 5
# BASE_FILENAME = DATA_NAME.lower()+"_"+str(CAPTIONS_PER_IMAGE)+"_cap_per_img_"+str(MIN_WORD_FREQ)+"_min_word_freq"
USE_HALF_PRECISION = True   
HALF_PRECISION_MODE = "O1"
FINE_TUNE_ENCODER = True
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
TRANSFORMS = transforms.Compose([normalize])


# ENCODER_STATE_FILE = DATA_FOLDER+"ENCODER_STATE_{}_".format(SEED)+DATA_NAME.upper()+".pt"
# DECODER_STATE_FILE = DATA_FOLDER+"DECODER_STATE_{}_".format(SEED)+DATA_NAME.upper()+".pt"
# ALL_CHECK_PT = DATA_FOLDER+"CHECKPOINT_{}_".format(SEED)+DATA_NAME+".pt"


PRETRAINED_EMBEDDINGS = "/mnt/BRCD-2/Pretrained/glove.840B.300d.txt"
PRETRAINED_EMBEDDINGS = None
FINE_TUNE_EMBEDDING = False
PRETRAINED_EMBEDDING_DIM = 300

if __name__ == "__main__":
    # SEED = 947
    # data_name = "FLICKR8K"
    
    # print("TRAINING ON {}  -  SEED: {}".format(data_name, SEED))

    
    # if SEED == 43:
    #     PRETRAINED_EMBEDDINGS = "/mnt/BRCD-2/Pretrained/glove.840B.300d.txt"
    #     FINE_TUNE_EMBEDDING = False
    #     PRETRAINED_EMBEDDING_DIM = 300

    # elif SEED == 947:
    #     PRETRAINED_EMBEDDINGS = "/mnt/BRCD-2/Pretrained/glove.840B.300d.txt"
    #     FINE_TUNE_EMBEDDING = True
    #     PRETRAINED_EMBEDDING_DIM = 300

    # elif SEED == 94743:
    #     PRETRAINED_EMBEDDINGS = None
    #     FINE_TUNE_EMBEDDING = False
    #     PRETRAINED_EMBEDDING_DIM = 512

    
    # train(SEED, DATA_FOLDER, data_name, CAPTIONS_PER_IMAGE, MIN_WORD_FREQ, PRETRAINED_EMBEDDINGS, PRETRAINED_EMBEDDING_DIM, FINE_TUNE_EMBEDDING, FINE_TUNE_ENCODER, USE_HALF_PRECISION, HALF_PRECISION_MODE, resume=False)
    
    
    # SEED = 94743
    # data_name = "FLICKR8K"
    # print("TRAINING ON {}  -  SEED: {}".format(data_name, SEED))
    
    # if SEED == 43:
    #     PRETRAINED_EMBEDDINGS = "/mnt/BRCD-2/Pretrained/glove.840B.300d.txt"
    #     FINE_TUNE_EMBEDDING = False
    #     PRETRAINED_EMBEDDING_DIM = 300

    # elif SEED == 947:
    #     PRETRAINED_EMBEDDINGS = "/mnt/BRCD-2/Pretrained/glove.840B.300d.txt"
    #     FINE_TUNE_EMBEDDING = True
    #     PRETRAINED_EMBEDDING_DIM = 300

    # elif SEED == 94743:
    #     PRETRAINED_EMBEDDINGS = None
    #     FINE_TUNE_EMBEDDING = False
    #     PRETRAINED_EMBEDDING_DIM = 512

    
    # train(SEED, DATA_FOLDER, data_name, CAPTIONS_PER_IMAGE, MIN_WORD_FREQ, PRETRAINED_EMBEDDINGS, PRETRAINED_EMBEDDING_DIM, FINE_TUNE_EMBEDDING, FINE_TUNE_ENCODER, USE_HALF_PRECISION, HALF_PRECISION_MODE, resume=False)
    

    SEED = 43
    data_name = "FLICKR30K"
    print("TRAINING ON {}  -  SEED: {}".format(data_name, SEED))
    
    if SEED == 43:
        PRETRAINED_EMBEDDINGS = "/mnt/BRCD-2/Pretrained/glove.840B.300d.txt"
        FINE_TUNE_EMBEDDING = False
        PRETRAINED_EMBEDDING_DIM = 300

    elif SEED == 947:
        PRETRAINED_EMBEDDINGS = "/mnt/BRCD-2/Pretrained/glove.840B.300d.txt"
        FINE_TUNE_EMBEDDING = True
        PRETRAINED_EMBEDDING_DIM = 300

    elif SEED == 94743:
        PRETRAINED_EMBEDDINGS = None
        FINE_TUNE_EMBEDDING = False
        PRETRAINED_EMBEDDING_DIM = 512

    
    train(SEED, DATA_FOLDER, data_name, CAPTIONS_PER_IMAGE, MIN_WORD_FREQ, PRETRAINED_EMBEDDINGS, PRETRAINED_EMBEDDING_DIM, FINE_TUNE_EMBEDDING, FINE_TUNE_ENCODER, USE_HALF_PRECISION, HALF_PRECISION_MODE, resume=False)


    SEED = 947
    data_name = "FLICKR30K"
    print("TRAINING ON {}  -  SEED: {}".format(data_name, SEED))
    
    if SEED == 43:
        PRETRAINED_EMBEDDINGS = "/mnt/BRCD-2/Pretrained/glove.840B.300d.txt"
        FINE_TUNE_EMBEDDING = False
        PRETRAINED_EMBEDDING_DIM = 300

    elif SEED == 947:
        PRETRAINED_EMBEDDINGS = "/mnt/BRCD-2/Pretrained/glove.840B.300d.txt"
        FINE_TUNE_EMBEDDING = True
        PRETRAINED_EMBEDDING_DIM = 300

    elif SEED == 94743:
        PRETRAINED_EMBEDDINGS = None
        FINE_TUNE_EMBEDDING = False
        PRETRAINED_EMBEDDING_DIM = 512

    
    train(SEED, DATA_FOLDER, data_name, CAPTIONS_PER_IMAGE, MIN_WORD_FREQ, PRETRAINED_EMBEDDINGS, PRETRAINED_EMBEDDING_DIM, FINE_TUNE_EMBEDDING, FINE_TUNE_ENCODER, USE_HALF_PRECISION, HALF_PRECISION_MODE, resume=False)
    
    SEED = 94743
    data_name = "FLICKR30K"
    print("TRAINING ON {}  -  SEED: {}".format(data_name, SEED))
    
    if SEED == 43:
        PRETRAINED_EMBEDDINGS = "/mnt/BRCD-2/Pretrained/glove.840B.300d.txt"
        FINE_TUNE_EMBEDDING = False
        PRETRAINED_EMBEDDING_DIM = 300

    elif SEED == 947:
        PRETRAINED_EMBEDDINGS = "/mnt/BRCD-2/Pretrained/glove.840B.300d.txt"
        FINE_TUNE_EMBEDDING = True
        PRETRAINED_EMBEDDING_DIM = 300

    elif SEED == 94743:
        PRETRAINED_EMBEDDINGS = None
        FINE_TUNE_EMBEDDING = False
        PRETRAINED_EMBEDDING_DIM = 512

    
    train(SEED, DATA_FOLDER, data_name, CAPTIONS_PER_IMAGE, MIN_WORD_FREQ, PRETRAINED_EMBEDDINGS, PRETRAINED_EMBEDDING_DIM, FINE_TUNE_EMBEDDING, FINE_TUNE_ENCODER, USE_HALF_PRECISION, HALF_PRECISION_MODE, resume=False)

    SEED = 43
    data_name = "COCO"
    print("TRAINING ON {}  -  SEED: {}".format(data_name, SEED))
    
    if SEED == 43:
        PRETRAINED_EMBEDDINGS = "/mnt/BRCD-2/Pretrained/glove.840B.300d.txt"
        FINE_TUNE_EMBEDDING = False
        PRETRAINED_EMBEDDING_DIM = 300

    elif SEED == 947:
        PRETRAINED_EMBEDDINGS = "/mnt/BRCD-2/Pretrained/glove.840B.300d.txt"
        FINE_TUNE_EMBEDDING = True
        PRETRAINED_EMBEDDING_DIM = 300

    elif SEED == 94743:
        PRETRAINED_EMBEDDINGS = None
        FINE_TUNE_EMBEDDING = False
        PRETRAINED_EMBEDDING_DIM = 512

    
    train(SEED, DATA_FOLDER, data_name, CAPTIONS_PER_IMAGE, MIN_WORD_FREQ, PRETRAINED_EMBEDDINGS, PRETRAINED_EMBEDDING_DIM, FINE_TUNE_EMBEDDING, FINE_TUNE_ENCODER, USE_HALF_PRECISION, HALF_PRECISION_MODE, resume=False)
    
    SEED = 947
    data_name = "COCO"
    print("TRAINING ON {}  -  SEED: {}".format(data_name, SEED))
    
    if SEED == 43:
        PRETRAINED_EMBEDDINGS = "/mnt/BRCD-2/Pretrained/glove.840B.300d.txt"
        FINE_TUNE_EMBEDDING = False
        PRETRAINED_EMBEDDING_DIM = 300

    elif SEED == 947:
        PRETRAINED_EMBEDDINGS = "/mnt/BRCD-2/Pretrained/glove.840B.300d.txt"
        FINE_TUNE_EMBEDDING = True
        PRETRAINED_EMBEDDING_DIM = 300

    elif SEED == 94743:
        PRETRAINED_EMBEDDINGS = None
        FINE_TUNE_EMBEDDING = False
        PRETRAINED_EMBEDDING_DIM = 512

    
    train(SEED, DATA_FOLDER, data_name, CAPTIONS_PER_IMAGE, MIN_WORD_FREQ, PRETRAINED_EMBEDDINGS, PRETRAINED_EMBEDDING_DIM, FINE_TUNE_EMBEDDING, FINE_TUNE_ENCODER, USE_HALF_PRECISION, HALF_PRECISION_MODE, resume=False)

    SEED = 94743
    data_name = "COCO"
    print("TRAINING ON {}  -  SEED: {}".format(data_name, SEED))
    
    if SEED == 43:
        PRETRAINED_EMBEDDINGS = "/mnt/BRCD-2/Pretrained/glove.840B.300d.txt"
        FINE_TUNE_EMBEDDING = False
        PRETRAINED_EMBEDDING_DIM = 300

    elif SEED == 947:
        PRETRAINED_EMBEDDINGS = "/mnt/BRCD-2/Pretrained/glove.840B.300d.txt"
        FINE_TUNE_EMBEDDING = True
        PRETRAINED_EMBEDDING_DIM = 300

    elif SEED == 94743:
        PRETRAINED_EMBEDDINGS = None
        FINE_TUNE_EMBEDDING = False
        PRETRAINED_EMBEDDING_DIM = 512

    
    train(SEED, DATA_FOLDER, data_name, CAPTIONS_PER_IMAGE, MIN_WORD_FREQ, PRETRAINED_EMBEDDINGS, PRETRAINED_EMBEDDING_DIM, FINE_TUNE_EMBEDDING, FINE_TUNE_ENCODER, USE_HALF_PRECISION, HALF_PRECISION_MODE, resume=False)

    # ENCODER_STATE_FILE = DATA_FOLDER+"ENCODER_STATE_{}_".format(SEED)+data_name.upper()+".pt"
    # DECODER_STATE_FILE = DATA_FOLDER+"DECODER_STATE_{}_".format(SEED)+data_name.upper()+".pt"
    # BASE_FILENAME = data_name.lower()+"_"+str(CAPTIONS_PER_IMAGE)+"_cap_per_img_"+str(MIN_WORD_FREQ)+"_min_word_freq"
    
    # word_map_file = os.path.join(DATA_FOLDER, 'WORDMAP_'+BASE_FILENAME+".json")
    # word_map = None
    # with open(word_map_file, 'r') as j:
    #     word_map  = json.load(j)
    # vocab_size = len(word_map)
                
    # encoder = Encoder(fine_tune=False)
    # decoder = Decoder(512, PRETRAINED_EMBEDDING_DIM ,512, vocab_size)
    
    # encoder.load_state_dict(torch.load(ENCODER_STATE_FILE))
    # decoder.load_state_dict(torch.load(DECODER_STATE_FILE))
    
    # dataset = ImageCaptionDataset(DATA_FOLDER, BASE_FILENAME, "VALID", TRANSFORMS)
    # bleu = None
    # for beam_size in [1,3,5,7]:
    #     bleu = evaluate(word_map, encoder, decoder, dataset, beam_size=beam_size, max_step=100)
    #     print(bleu)
    #     for j, b in enumerate(bleu):
    #         ALL_STATS.loc[i, "Valid-BLEU-"+str(j)] = b
    
    
    # dataset = ImageCaptionDataset(DATA_FOLDER, BASE_FILENAME, "TEST", TRANSFORMS)
    # bleu = None
    # for beam_size in [1,3,5,7]:
    #     bleu = evaluate(word_map, encoder, decoder, dataset, beam_size=beam_size, max_step=100)
    #     print(bleu)
    #     for j, b in enumerate(bleu):
    #         ALL_STATS.loc[i, "Test-BLEU-"+str(j)] = b
    
    # ALL_STATS.to_csv("ALL_STATS.csv", index=False)
    # i+=1

