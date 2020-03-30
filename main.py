from train import train
from data import ImageCaptionDataset
from models import Encoder, Decoder
from torch.utils.data import DataLoader
import json, os

SEED = 43
DATA_FOLDER = "./output/"
DATA_NAME = "COCO"
CAPTIONS_PER_IMAGE = 5
MIN_WORD_FREQ = 5
BASE_FILENAME = DATA_NAME.lower()+"_"+str(CAPTIONS_PER_IMAGE)+"_cap_per_img_"+str(MIN_WORD_FREQ)+"_min_word_freq"
USE_HALF_PRECISION = True   
HALF_PRECISION_MODE = "O1"
FINE_TUNE_ENCODER = True

if __name__ == "__main__":
    
    train(SEED, DATA_FOLDER, DATA_NAME, CAPTIONS_PER_IMAGE, MIN_WORD_FREQ,FINE_TUNE_ENCODER, USE_HALF_PRECISION, HALF_PRECISION_MODE, resume=True)

    # train(DATA_FOLDER, DATA_NAME, CAPTIONS_PER_IMAGE, MIN_WORD_FREQ)
    