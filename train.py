import time
import torch.backends.cudnn as cudnn
import torch
from data import ImageCaptionDataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from commons import *
import os, json, tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from models import Decoder, Encoder

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

def adjust_learning_rate(optimizer, scale):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*scale
    return optimizer

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']/2.0

def clear_cuda():
    torch.cuda.empty_cache()
    gc.collect()

def print_epoch_stat(epoch_idx, time_elapsed_in_seconds, history=None, train_loss=None, train_accuracy=None, valid_loss=None, valid_accuracy=None):
    print("\n\nEPOCH {} Completed, Time Taken: {}".format(epoch_idx+1, datetime.timedelta(seconds=time_elapsed_in_seconds)))
    if train_loss is not None:
        if history is not None:
            history.loc[epoch_idx, "train_loss"] = train_loss
        print("\tTrain Loss \t{:0.9}".format(train_loss))
    if train_accuracy is not None:
        if history is not None:
            history.loc[epoch_idx, "train_accuracy"] = 100.0*train_accuracy
        print("\tTrain Accuracy \t{:0.9}%".format(100.0*train_accuracy))
    if valid_loss is not None:
        if history is not None:
            history.loc[epoch_idx, "valid_loss"] = valid_loss
        print("\tValid Loss \t{:0.9}".format(valid_loss))
    if valid_accuracy is not None:
        if history is not None:
            history.loc[epoch_idx, "valid_accuracy"] = 100.0*valid_accuracy
        print("\tValid Accuracy \t{:0.9}%".format(100.0*valid_accuracy))

    return history

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    """Coding Credit: https://github.com/Bjarten/early-stopping-pytorch"""
    def __init__(self, track="min", patience=7, verbose=False, delta=0, save_model_name="checkpoint.pt"):
        """
        Args:
            track (str): What to track, possible value: min, max (e.g. min validation loss, max validation accuracy (%))
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.track = track
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_best = -np.Inf
        if self.track=="min":
            self.val_best = np.Inf
        self.delta = delta
        self.save_model_name = save_model_name

    def __call__(self, current_score, model):

        score = -current_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(current_score, model)
        elif ((score < self.best_score + self.delta) and self.track=="min") or ((score>self.best_score+self.delta) and self.track=="max"):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(current_score, model)
            self.counter = 0

    def save_checkpoint(self, new_best, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Found better solution ({self.val_best:.6f} --> {new_best:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_model_name)
        self.val_best = new_best














def train(data_folder, data_name, captions_per_image, min_word_freq, resume=False):
    data_name = data_name.lower()
    assert data_name in {"coco", 'flickr8k', 'flickr30k'}
    
    base_filename = data_name+"_"+str(captions_per_image)+"_cap_per_img_"+str(min_word_freq)+"_min_word_freq"
    cudnn.benchmark = True
    
    
    # MODEL PARAMS
    emb_dim = 512
    attention_dim = 512
    decoder_dim = 512
    dropout = 0.5
    
    
    # TRAINING PARAMS
    start_epoch = 0
    epochs = 120
    batch_size = 64
    early_stopper = EarlyStopping(track="min", save_model_name=os.path.join(data_folder,"BEST_MODEL_SO_FAR.pt"))
    workers = 1
    encoder_lr = 1e-4
    decoder_lr = 5e-4
    grad_clip = 5.0
    alpha_c = 1.0
    best_bleu4 = 0.0
    fine_tune_encoder = False
    checkpoint = "checkpoint.pt"
    
    
    word_map_file = os.path.join(data_folder, 'WORDMAP_'+base_filename+".json")
    word_map = None
    with open(word_map_file, 'r') as j:
        word_map  = json.load(j)
    
    
    
    decoder = Decoder(attention_dim=attention_dim, embedding_dim=emb_dim, decoder_dim=decoder_dim, vocab_size=len(word_map), dropout=0.5)
    encoder = Encoder()
    encoder.fine_tune(fine_tune_encoder)
    
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=decoder_lr)
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()), lr=encoder_lr) if fine_tune_encoder else None
    
    
    encoder.to(device)
    decoder.to(device)
    
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_dataset = ImageCaptionDataset(data_folder, base_filename, "train", transform=transforms.Compose([normalize]))
    valid_dataset = ImageCaptionDataset(data_folder, base_filename, "valid", transform=transforms.Compose([normalize]))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=workers, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=workers, pin_memory=True, shuffle=False)
    
    
    train_single_epoch(encoder, decoder, train_loader, encoder_optimizer, decoder_optimizer, criterion)
    
    
    
    
    
    if resume:
        pass
    else:
        pass



def train_single_epoch(encoder, decoder, train_loader, encoder_optimizer, decoder_optimizer, criterion, alpha_c=1.0, grad_clip=1.0):
    tl, ta = 0, 0 
    encoder.train()
    decoder.train()
    
    for i, (imgs, caps, caplens) in enumerate(tqdm.tqdm(train_loader, total=len(train_loader))):
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        
        
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
        
        targets = caps_sorted[:, 1:]
        
        scores = torch.nn.utils.rnn.pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = torch.nn.utils.rnn.pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        
        loss = criterion(scores, targets)
        
        loss+=alpha_c*((1.0 - alphas.sum(dim=1))**2).mean()
        
        ta+= calculate_accuracy(scores, targets, 5)
        tl+=loss.item()
        
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        
        loss.backward()
        
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
            if encoder_optimizer is not None:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
        
        
        decoder_optimizer.step()
        
        if encoder_optimizer is not None:
            encoder_optimizer.step()
        
        if i%50 == 0:
            print("TL: {}, TA: {}".format(1.0*tl/(i+1), 1.0*ta/(i+1)))

    return tl/len(train_loader), ta/len(train_loader)