from train import train

DATA_FOLDER = "./output/"
DATA_NAME = "COCO"
CAPTIONS_PER_IMAGE = 5
MIN_WORD_FREQ = 5



if __name__ == "__main__":
    train(DATA_FOLDER, DATA_NAME, CAPTIONS_PER_IMAGE, MIN_WORD_FREQ)
    