from train import train
from data import ImageCaptionDataset
from models import Encoder, Decoder
from torch.utils.data import DataLoader
import torch, time, datetime
import pandas as pd
import json, os, tqdm, sys
from eval import evaluate, generate_caption
from torchvision.transforms import transforms
from glob import glob
from train import clear_cuda
from commons import *


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
FINE_TUNE_EMBEDDING = False

FLICKR30K_DEMO_PATH = "./demo/flickr30k/"
FLICKR8K_DEMO_PATH = "./demo/flickr8k/"
COCO_DEMO_PATH = "./demo/coco/"

STATS_PATH = "./stats/"


def generate_stats(SEED, data_name, PRETRAINED_EMBEDDING_DIM, BEAM_SIZES = [1,3,5,7], max_step=200):
    ENCODER_STATE_FILE = DATA_FOLDER+"ENCODER_STATE_{}_".format(SEED)+data_name.upper()+".pt"
    DECODER_STATE_FILE = DATA_FOLDER+"DECODER_STATE_{}_".format(SEED)+data_name.upper()+".pt"
    BASE_FILENAME = data_name.lower()+"_"+str(CAPTIONS_PER_IMAGE)+"_cap_per_img_"+str(MIN_WORD_FREQ)+"_min_word_freq"
    
    word_map_file = os.path.join(DATA_FOLDER, 'WORDMAP_'+BASE_FILENAME+".json")
    word_map = None
    with open(word_map_file, 'r') as j:
        word_map  = json.load(j)
    vocab_size = len(word_map)
                
    encoder = Encoder(fine_tune=False)
    decoder = Decoder(512, PRETRAINED_EMBEDDING_DIM ,512, vocab_size)
    
    encoder.load_state_dict(torch.load(ENCODER_STATE_FILE))
    decoder.load_state_dict(torch.load(DECODER_STATE_FILE))
    
    df_save_name = os.path.join(STATS_PATH, '{}_{}_STATS.csv'.format(data_name.upper(), SEED))
    df = pd.DataFrame()
    idx = 0
        
    dataset = ImageCaptionDataset(DATA_FOLDER, BASE_FILENAME, "VALID", TRANSFORMS)
    for beam_size in BEAM_SIZES:
        df.loc[idx, "BEAM_SIZE"] = beam_size
        df.loc[idx, "DATA-SPLIT"] = "VALID"
        all_bleu, meteor_score, rouge_score, cider_score, spice_score = evaluate(word_map, encoder, decoder, dataset, beam_size=beam_size, max_step=max_step)
        
        print("BLEU: ", all_bleu)
        print("METEOR: ",meteor_score)
        print("ROGUE: ",rouge_score)
        print("CIDEr: ", cider_score)
        print("SPICE: ", spice_score)

        for j, b in enumerate(all_bleu):
            df.loc[idx, "BLEU-"+str(j+1)] = b
        df.loc[idx, "METEOR"] = meteor_score
        df.loc[idx, "ROUGE-L"] = rouge_score
        df.loc[idx, "CIDEr"] = cider_score
        df.loc[idx, "SPICE"] = spice_score
        idx+=1
            
    dataset = ImageCaptionDataset(DATA_FOLDER, BASE_FILENAME, "TEST", TRANSFORMS)
    
    for beam_size in BEAM_SIZES:
        df.loc[idx, "BEAM_SIZE"] = beam_size
        df.loc[idx, "DATA-SPLIT"] = "TEST"
        all_bleu, meteor_score, rouge_score, cider_score, spice_score = evaluate(word_map, encoder, decoder, dataset, beam_size=beam_size, max_step=max_step)
        
        print("BLEU: ", all_bleu)
        print("METEOR: ",meteor_score)
        print("ROGUE: ",rouge_score)
        print("CIDEr: ", cider_score)
        print("SPICE: ", spice_score)

        for j, b in enumerate(all_bleu):
            df.loc[idx, "BLEU-"+str(j+1)] = b
        df.loc[idx, "METEOR"] = meteor_score
        df.loc[idx, "ROUGE-L"] = rouge_score
        df.loc[idx, "CIDEr"] = cider_score
        df.loc[idx, "SPICE"] = spice_score
        idx+=1
             
    df.to_csv(df_save_name, index=False)
    
    del encoder
    del decoder
    del dataset
    del df




def generate_captions(SEED, image_paths, data_name, PRETRAINED_EMBEDDING_DIM, BEAM_SIZE = 5, max_step=200):
    ENCODER_STATE_FILE = DATA_FOLDER+"ENCODER_STATE_{}_".format(SEED)+data_name.upper()+".pt"
    DECODER_STATE_FILE = DATA_FOLDER+"DECODER_STATE_{}_".format(SEED)+data_name.upper()+".pt"
    BASE_FILENAME = data_name.lower()+"_"+str(CAPTIONS_PER_IMAGE)+"_cap_per_img_"+str(MIN_WORD_FREQ)+"_min_word_freq"
    
    word_map_file = os.path.join(DATA_FOLDER, 'WORDMAP_'+BASE_FILENAME+".json")
    word_map = None
    with open(word_map_file, 'r') as j:
        word_map  = json.load(j)
    vocab_size = len(word_map)
                
    encoder = Encoder(fine_tune=False)
    decoder = Decoder(512, PRETRAINED_EMBEDDING_DIM ,512, vocab_size)
    
    encoder.load_state_dict(torch.load(ENCODER_STATE_FILE))
    decoder.load_state_dict(torch.load(DECODER_STATE_FILE))
    
    all_images_jpg = glob(image_paths+"*.jpg")
    all_images_png = glob(image_paths+"*.png")
    
    all_images = all_images_jpg+all_images_png
    
    for image_path in tqdm.tqdm(all_images): 
        generate_caption(SEED,encoder, decoder, image_path, word_map, BEAM_SIZE, max_step)


if __name__ == "__main__":

    try:
        
        assert len(sys.argv) == 4
        
        print(sys.argv)
        
        SEED = int(sys.argv[1])
        data_name = sys.argv[2]
        resume = sys.argv[3].lower() == "true" 
        
        print("RESUME?? -->", resume)
        print("USING {}  -  SEED: {}".format(data_name, SEED))

        
        if SEED == 43:
            PRETRAINED_EMBEDDINGS = "/mnt/BRCD-2/Pretrained/glove.840B.300d.txt"
            FINE_TUNE_EMBEDDING = False
            PRETRAINED_EMBEDDING_DIM = 300

        elif SEED == 947:
            PRETRAINED_EMBEDDINGS = "/mnt/BRCD-2/Pretrained/glove.840B.300d.txt"
            FINE_TUNE_EMBEDDING = True
            PRETRAINED_EMBEDDING_DIM = 300

        elif SEED == 94743 :
            PRETRAINED_EMBEDDINGS = None
            FINE_TUNE_EMBEDDING = False
            PRETRAINED_EMBEDDING_DIM = 512

        train(SEED, DATA_FOLDER, data_name, CAPTIONS_PER_IMAGE, MIN_WORD_FREQ, PRETRAINED_EMBEDDINGS, FINE_TUNE_EMBEDDING, FINE_TUNE_ENCODER, USE_HALF_PRECISION, HALF_PRECISION_MODE, resume=resume)
        generate_stats(SEED, data_name, PRETRAINED_EMBEDDING_DIM)
        # generate_captions(SEED, "./output/very_small_test/",data_name, PRETRAINED_EMBEDDING_DIM) 
        
        
    except (Exception, KeyboardInterrupt) as e:
        print(e)
        mode = False
        if VISDOM.win_exists(win="ERROR"):
            mode = True
        VISDOM.text("CRASHED or Closed at "+str(datetime.datetime.fromtimestamp(time.time())), win="ERROR", append=mode, opts={'title':"Errors"})
        raise e

    VISDOM.text("Training Completed at "+str(datetime.datetime.fromtimestamp(time.time())), win="FINISHED", opts={'title':"FINISHED"})