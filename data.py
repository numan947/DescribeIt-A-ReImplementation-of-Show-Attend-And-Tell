import json, os, random, h5py, tqdm
from collections import Counter
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch

def parse_and_prepare_data(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder, max_len):
    dataset = dataset.lower()
    assert dataset in {'coco', 'flickr8k', 'flickr30k'}
    
    with open(karpathy_json_path, 'r') as j:
        karpathy_data = json.load(j)
    
    trn_image_paths = []
    vld_image_paths = []
    tst_image_paths = []
    trn_image_captions = []
    vld_image_captions = []
    tst_image_captions = []
    
    word_freq = Counter()
    
    print("READING CAPTIONS.....")
    for img in tqdm.tqdm(karpathy_data["images"]):
        captions = []
        for c in img['sentences']:
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue
            
        
        path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(image_folder, img['filename'])
        
        
        if img['split'] in {'train', 'restval'}:
            trn_image_paths.append(path)
            trn_image_captions.append(captions)
        
        elif img['split'] in {'val'}:
            vld_image_captions.append(captions)
            vld_image_paths.append(path)
        elif img['split'] in {'test'}:
            tst_image_paths.append(path)
            tst_image_captions.append(captions)
    
    
    assert len(trn_image_captions) == len(trn_image_paths)
    assert len(vld_image_captions) == len(vld_image_paths)
    assert len(tst_image_captions) == len(tst_image_paths)
    
    words = [w for w in word_freq.keys() if word_freq[w]>min_word_freq]
    word_map = {k:v+1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map)+1
    word_map['<start>'] = len(word_map)+1
    word_map['<end>'] = len(word_map)+1
    word_map['<pad>'] = 0
    
    
    base_filename = dataset+"_"+str(captions_per_image)+"_cap_per_img_"+str(min_word_freq)+"_min_word_freq"
    
    
    with open(os.path.join(output_folder, "WORDMAP_"+base_filename+".json"), 'w') as j:
        json.dump(word_map, j)
    
    
    random.seed(947)
    
    for img, cap, split in [(trn_image_paths, trn_image_captions, "TRAIN"), (vld_image_paths, vld_image_captions, "VALID"), (tst_image_paths, tst_image_captions, "TEST")]:
        
        with h5py.File(os.path.join(output_folder, split+"_IMAGES_"+base_filename+".hdf5"), 'a') as h:
            h.attrs['captions_per_image'] = captions_per_image
            images = h.create_dataset('images', (len(img), 3, 256, 256), dtype='uint8')
            
            print("\nReading %s images and captions, storing to file ....\n"%split)
            
            enc_captions = []
            caplens = []
            
            for i, path in enumerate(tqdm.tqdm(img)):
                
                if len(cap[i]) < captions_per_image:
                    captions = cap[i]+[random.choice(cap[i]) for _ in range(captions_per_image - len(cap[i]))]
                else:
                    captions = random.sample(cap[i], k=captions_per_image)
                
                assert len(captions) == captions_per_image
                
                
                image = Image.open(img[i])
                image = image.convert("RGB")
                image = image.resize((256,256))
                image = np.transpose(np.array(image), (2,0,1))
                assert image.shape == (3,256,256)
                assert np.max(image)<=255

                images[i] = image
                
                for j, c in enumerate(captions):
                    enc_c = [word_map['<start>']]+[word_map.get(word, word_map['<unk>']) for word in c]+[word_map['<end>']]+[word_map['<pad>']]*(max_len-len(c))
                    c_len = len(c)+2
                    
                    enc_captions.append(enc_c)
                    caplens.append(c_len)
            
            assert images.shape[0]*captions_per_image == len(enc_captions) == len(caplens)
            
            
            with open(os.path.join(output_folder, split+'_CAPTIONS_'+base_filename+'.json'), 'w') as j:
                json.dump(enc_captions, j)
                
            with open(os.path.join(output_folder, split+'_CAPLENS_'+base_filename+'.json'), 'w') as j:
                json.dump(caplens, j)



class ImageCaptionDataset(Dataset):
    def __init__(self, data_folder, base_filename, split, transform=None):
        self.split = split.upper()
        assert self.split in {"TRAIN", "TEST", "VALID"}
        
        
        self.h = h5py.File(os.path.join(data_folder, self.split+"_IMAGES_"+base_filename+".hdf5"), 'r')
        
        self.images = self.h['images']
        
        self.cpi = self.h.attrs['captions_per_image']
        
        with open(os.path.join(data_folder, self.split+"_CAPTIONS_"+base_filename+".json"), 'r') as j:
            self.captions = json.load(j)
            
        with open(os.path.join(data_folder, self.split+"_CAPLENS_"+base_filename+".json"), 'r') as j:
            self.caplens = json.load(j)
        
        self.transform = transform
        
        self.dataset_size = len(self.captions)
    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, i):
        img = torch.FloatTensor(self.images[i//self.cpi]/255.0)
        if self.transform is not None:
            img = self.transform(img)
        
        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])
        
        if self.split == "TRAIN":
            return img, caption, caplen
        else:
            all_captions = torch.LongTensor(
                self.captions[((i//self.cpi)*self.cpi):(((i//self.cpi)*self.cpi)+self.cpi)]
            )
            return img, caption, caplen, all_captions

if __name__ == "__main__":
    pass
    # data = ImageCaptionDataset("./output", "")
# This is for testing the code in this file and creating the dataset for the first time 
    # parse_and_prepare_data('coco',
    #                        '/mnt/BRCD-2/Datasets/karpathy_captions/dataset_coco.json',
    #                        '/mnt/BRCD-2/Datasets/coco',
    #                        5,
    #                        5,
    #                        './output/',
    #                        50
    #                        )
    # parse_and_prepare_data('flickr8k',
    #                        '/mnt/BRCD-2/Datasets/karpathy_captions/dataset_flickr8k.json',
    #                        '/mnt/BRCD-2/Datasets/flickr8k/flickr8k_images',
    #                        5,
    #                        5,
    #                        './output/',
    #                        50
    #                        )
    # parse_and_prepare_data('flickr30k',
    #                        '/mnt/BRCD-2/Datasets/karpathy_captions/dataset_flickr30k.json',
    #                        '/mnt/BRCD-2/Datasets/flickr30k/flickr30k_images',
    #                        5,
    #                        5,
    #                        './output/',
    #                        50
    #                        )