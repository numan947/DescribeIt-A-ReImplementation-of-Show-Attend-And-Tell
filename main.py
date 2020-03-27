from train import train
from data import ImageCaptionDataset
from models import Encoder, Decoder
from torch.utils.data import DataLoader
import json, os

DATA_FOLDER = "./output/"
DATA_NAME = "COCO"
CAPTIONS_PER_IMAGE = 5
MIN_WORD_FREQ = 5
BASE_FILENAME = DATA_NAME.lower()+"_"+str(CAPTIONS_PER_IMAGE)+"_cap_per_img_"+str(MIN_WORD_FREQ)+"_min_word_freq"


if __name__ == "__main__":
    
    train(DATA_FOLDER, DATA_NAME, CAPTIONS_PER_IMAGE, MIN_WORD_FREQ)

    # train(DATA_FOLDER, DATA_NAME, CAPTIONS_PER_IMAGE, MIN_WORD_FREQ)
    