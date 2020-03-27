import time
import torch.backends.cudnn as cudnn
import torch
from data import ImageCaptionDataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from commons import *
import os, json, tqdm, datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from models import Decoder, Encoder
import apex, gc
from apex import amp
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score

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
        return param_group['lr']

def clear_cuda():
    torch.cuda.empty_cache()
    gc.collect()

def print_epoch_stat(epoch_idx, time_elapsed_in_seconds, history=None, train_loss=None, train_accuracy=None, valid_loss=None, valid_accuracy=None, bleu4=None):
    print("\n\nEPOCH {} Completed, Time Taken: {}".format(epoch_idx+1, datetime.timedelta(seconds=time_elapsed_in_seconds)))
    if train_loss is not None:
        if history is not None:
            history.loc[epoch_idx, "train_loss"] = train_loss
        print("\tTrain Loss \t{:0.9}".format(train_loss))
    if train_accuracy is not None:
        if history is not None:
            history.loc[epoch_idx, "train_accuracy"] = train_accuracy
        print("\tTrain Accuracy \t{:0.9}%".format(100.0*train_accuracy))
    if valid_loss is not None:
        if history is not None:
            history.loc[epoch_idx, "valid_loss"] = valid_loss
        print("\tValid Loss \t{:0.9}".format(valid_loss))
    if valid_accuracy is not None:
        if history is not None:
            history.loc[epoch_idx, "valid_accuracy"] = valid_accuracy
        print("\tValid Accuracy \t{:0.9}%".format(100.0*valid_accuracy))
    if bleu4 is not None:
        if history is not None:
            history.loc[epoch_idx, "bleu4"] = bleu4
        print("\tBLEU Score \t{:0.9}%".format(100.0*bleu4))

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














def train(seed, data_folder, data_name, captions_per_image, min_word_freq, use_half_precision=False, half_precision_mode=None, half_precision_loss_scale="dynamic", resume=False):
    clear_cuda()
    torch.manual_seed(seed)
    
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
    early_stopper = EarlyStopping(track="min", save_model_name=os.path.join(data_folder,"BEST_MODEL_SO_FAR.pt"), patience=12)
    workers = 1
    encoder_lr = 1e-4
    decoder_lr = 5e-4
    grad_clip = 5.0
    alpha_c = 1.0
    best_bleu4 = 0.0
    fine_tune_encoder = True
    checkpoint = "checkpoint.pt"
    history = "SEED_{}_HISTORY.csv".format(seed)
    
    
    
    
    checkpoint = os.path.join(data_folder, checkpoint)
    history_path = os.path.join(data_folder, history)
    word_map_file = os.path.join(data_folder, 'WORDMAP_'+base_filename+".json")
    word_map = None
    with open(word_map_file, 'r') as j:
        word_map  = json.load(j)
    
    
    
    decoder = Decoder(attention_dim=attention_dim, embedding_dim=emb_dim, decoder_dim=decoder_dim, vocab_size=len(word_map), dropout=0.5)
    encoder = Encoder()
    encoder.fine_tune(fine_tune_encoder)
    
    if not use_half_precision:
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=decoder_lr)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()), lr=encoder_lr) if fine_tune_encoder else None
    else:
        decoder_optimizer = apex.optimizers.FusedAdam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=decoder_lr)
        encoder_optimizer = apex.optimizers.FusedAdam(params=filter(lambda p: p.requires_grad, encoder.parameters()), lr=encoder_lr) if fine_tune_encoder else None
    
    
    encoder.to(device)
    decoder.to(device)
    
    if use_half_precision:
        if encoder_optimizer is not None:
            [decoder, encoder], [decoder_optimizer, encoder_optimizer] = amp.initialize([decoder, encoder], [decoder_optimizer,encoder_optimizer], opt_level=half_precision_mode, loss_scale=half_precision_loss_scale)
        else:
            decoder, decoder_optimizer = amp.initialize(decoder, decoder_optimizer, opt_level=half_precision_mode, loss_scale=half_precision_loss_scale)
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_dataset = ImageCaptionDataset(data_folder, base_filename, "train", transform=transforms.Compose([normalize]))
    valid_dataset = ImageCaptionDataset(data_folder, base_filename, "valid", transform=transforms.Compose([normalize]))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=workers, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=workers, pin_memory=True, shuffle=False)
    
    
    if resume:
        history = pd.read_csv(history_path, index_col="Epoch")
        saved_information = torch.load(checkpoint, map_location=torch.device('cpu'))
        
        encoder.load_state_dict(saved_information["encoder_state"])
        if encoder_optimizer is not None:
            encoder_optimizer.load_state_dict(saved_information['encoder_optimizer_state'])
        
        decoder.load_state_dict(saved_information['decoder_state'])
        decoder_optimizer.load_state_dict(saved_information['decoder_optimizer_state'])
        
        amp.load_state_dict(saved_information['amp_state'])
        
        start_epoch = saved_information['epoch'] + 1
        early_stopper.counter = saved_information['early_stopper_counter']
        early_stopper.early_stop = saved_information['early_stopper_early_stop']
        print("Resuming From {}".format(start_epoch))
        
    else:
        history = pd.DataFrame()
        history.index.name = "Epoch"
    
    
    for e in range(start_epoch, epochs):
        if early_stopper.early_stop:
            break
        
        if early_stopper.counter>0 and early_stopper.counter%8==0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)
        
        curt = time.time()    
        print("EPOCH: {}/{}, Encoder LR: {}, Decoder LR: {}".format(e, epochs, get_lr(encoder_optimizer) if fine_tune_encoder else -1000, get_lr(decoder_optimizer)))
        tl, ta = train_single_epoch(encoder, decoder, train_loader, encoder_optimizer, decoder_optimizer, criterion, use_half_precision=use_half_precision)
        vl, va, bleu4 = evaluate_on_validation(encoder, decoder, criterion, valid_loader, word_map)
        early_stopper(bleu4, decoder)
        print_epoch_stat(e, time.time()-curt, history, tl, ta, vl, va, bleu4)
        print("SAVING ... ", end="")
        save_information = {
            'epoch':e,
            'encoder_optimizer_state':encoder_optimizer.state_dict() if fine_tune_encoder else None,
            'decoder_optimizer_state':decoder_optimizer.state_dict(),
            'amp_state':amp.state_dict() if use_half_precision else None,
            'encoder_state':encoder.state_dict(),
            'decoder_state':decoder.state_dict(),
            'early_stopper_counter':early_stopper.counter,
            'early_stopper_early_stop':early_stopper.early_stop
        }
        
        torch.save(save_information, checkpoint)
        history.to_csv(history_path, index="Epoch")
        print("SAVED")

def train_single_epoch(encoder, decoder, train_loader, encoder_optimizer, decoder_optimizer, criterion, alpha_c=1.0, grad_clip=0.97, use_half_precision=False):
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
        
        if use_half_precision:
            if encoder_optimizer is not None:
                with amp.scale_loss(loss, [decoder_optimizer, encoder_optimizer]) as scaled_loss:
                    scaled_loss.backward()
            else:
                with amp.scale_loss(loss, decoder_optimizer) as scaled_loss:
                    scaled_loss.backward()
        else:
            loss.backward()
        
        if grad_clip is not None:
            if not use_half_precision:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
                if encoder_optimizer is not None:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(amp.master_params(decoder_optimizer), grad_clip)
                if encoder_optimizer is not None:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(encoder_optimizer), grad_clip)


        decoder_optimizer.step()
        
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        if i>0 and i%2000 == 0:
            print("TL: {}, TA: {}".format(1.0*tl/(i+1), 1.0*ta/(i+1)))
    return tl/len(train_loader), ta/len(train_loader)

def evaluate_on_validation(encoder, decoder, criterion, valid_loader, word_map, alpha_c=1.0):
    vl, va = 0.0,0.0
    
    encoder.eval()
    decoder.eval()

    references = list()
    hypotheses = list()

    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(tqdm.tqdm(valid_loader, total=len(valid_loader))):
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            
            imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
            
            targets = caps_sorted[:, 1:]
            
            scores_copy = scores.clone()
            
            scores = torch.nn.utils.rnn.pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = torch.nn.utils.rnn.pack_padded_sequence(targets, decode_lengths, batch_first=True).data
            
            loss = criterion(scores, targets)
            loss = loss + alpha_c*((1.0-alphas.sum(dim=1))**2).mean()
            vl+=loss.item()
            va+=calculate_accuracy(scores, targets, k=5)
            
                        
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(map(lambda  c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}], img_caps))
                
                references.append(img_captions)
            
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])
            preds = temp_preds
            hypotheses.extend(preds)
            assert len(references) == len(hypotheses)
            
    return vl/len(valid_loader), va/len(valid_loader), corpus_bleu(references, hypotheses)