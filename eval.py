from commons import *
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import tqdm, os
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from PIL import Image

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform as trxf



def beam_search(encoder, decoder, image, word_map, beam_size, max_step):
    vocab_size = len(word_map)
    k = beam_size
    image = image.to(device)
    encoder_out = encoder(image)
    encoder_dim = encoder_out.size(-1)
    encoder_image_size = encoder_out.size(1)
    encoder_out = encoder_out.view(1, -1, encoder_dim)
    num_pixels = encoder_out.size(1)
    
    # treat k as batch size
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)
    
    # k prev words
    k_prev_words = torch.LongTensor([[word_map['<start>']] * k]).to(device)
    k_prev_words = k_prev_words.view(k, -1)
    
    seqs = k_prev_words
    top_k_scores = torch.zeros(k, 1).to(device)
    seq_alpha = torch.ones(k, 1, encoder_image_size, encoder_image_size).to(device)
    
    completed_seqs = list()
    completed_seqs_alpha = list()
    completed_seqs_scores = list()
    
    
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)
    
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)
        
        attended_encoder_out, attentions = decoder.attention(encoder_out, h)
        
        attentions = attentions.view(-1, encoder_image_size, encoder_image_size)
        
        
        gate = torch.sigmoid(decoder.f_beta(h))
        attended_encoder_out = gate * attended_encoder_out

        
        h,c = decoder.decode_step(torch.cat([embeddings, attended_encoder_out], dim=1), (h, c))
        
        scores = torch.log_softmax(decoder.fc(h), dim=1)
        
        scores = top_k_scores.expand_as(scores) + scores
        
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)
        
        
        prev_word_inds = top_k_words/vocab_size
        next_word_inds = top_k_words%vocab_size
        
        
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
        seq_alpha = torch.cat([seq_alpha[prev_word_inds], attentions[prev_word_inds].unsqueeze(1)], dim=1)
        
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word!=word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
        
        if len(complete_inds)>0:
            completed_seqs.extend(seqs[complete_inds].tolist())
            completed_seqs_alpha.extend(seq_alpha[complete_inds].tolist())
            completed_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)
            if k<=0:
                break
        
        seqs = seqs[incomplete_inds]
        seq_alpha = seq_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
        
        if step>max_step:
            break
        step+=1
    return completed_seqs, completed_seqs_scores, completed_seqs_alpha
    



def evaluate(word_map, encoder, decoder, dataset, beam_size, max_step=50):
    
    rev_wordmap = {v:k for k,v in word_map.items()}
    
    BATCH_SIZE = 1 # MUST BE 1

    NUM_WORKERS = 1
    SHUFFLE = False
    PIN_MEMORY = True
    
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY, shuffle=SHUFFLE, num_workers=NUM_WORKERS)
    
    references = list()
    hypotheses = list() 
    
    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(tqdm.tqdm(loader, desc="EVALUATING AT BEAM SIZE {}".format(beam_size), total=len(loader))):
            completed_seqs, completed_seqs_scores, completed_seqs_alpha = beam_search(encoder, decoder, imgs, word_map, beam_size=beam_size, max_step=max_step)
            try:
                i = completed_seqs_scores.index(max(completed_seqs_scores))
                seq = completed_seqs[i]
                img_caps = allcaps.squeeze(0).tolist()
                
                img_captions = list(map(lambda c: [w for w in c if w not in {word_map['<start>'],word_map['<end>'],word_map['<pad>']}], img_caps))
                references.append(img_captions)
                hypotheses.append([w for w in seq if w not in {word_map['<start>'],word_map['<end>'],word_map['<pad>']}])
                
                assert len(hypotheses) == len(references)
                
            except Exception as  e:
                print("CAPS GENERATION FAILED FOR {}th image".format(i))
        
        all_bleu = list()
        
        for i in range(1, 6):
            curb = i
            all_bleu.append(corpus_bleu(references, hypotheses, weights=tuple([1.0/float(curb) for _ in range(i)])))
        
        
        return all_bleu


def visualize_attention(image_path, seq, alphas, rev_wordmap, smooth=True, max_step=50):
    save_path = os.path.abspath(image_path).split(".")[0]+"_ANNOTATED.jpg"
    UPSAMPLE_SIZE = 25
    image = Image.open(image_path)
    image = image.resize([14*UPSAMPLE_SIZE,14*UPSAMPLE_SIZE], Image.LANCZOS)
    words = [rev_wordmap[p] for p in seq]
    
    for t in range(len(words)):
        if t>max_step:
            break
        plt.subplot(np.ceil(len(words)/5.0), 5, t+1)
        plt.text(0, 1, '%s'%(words[t]), color='black', backgroundcolor='white', fontsize=10)
        plt.imshow(image)
        
        current_alpha = alphas[t,:]
        
        if smooth:
            alpha = trxf.pyramid_expand(current_alpha.numpy(), upscale=UPSAMPLE_SIZE, sigma=8)
        else:
            alpha = trxf.resize(current_alpha.numpy(), [14*UPSAMPLE_SIZE, 14*UPSAMPLE_SIZE])
        
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.75)
        
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(' '.join(words))

def generate_caption(encoder, decoder, image_paths, word_map, beam_size, max_step=50):
    if not isinstance(image_paths, list):
        image_paths = [image_paths]
    
    rev_wmap = {v:k for k,v in word_map.items()}
    txf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()
        
    for image_path in image_paths:        
        img = Image.open(image_path)
        img = img.convert("RGB").resize((256,256))
        img = txf(img).unsqueeze(0)        
        img = img.to(device)
        with torch.no_grad():
            completed_seqs, completed_seqs_scores, completed_seqs_alpha = beam_search(encoder, decoder, img, word_map, beam_size, max_step)
            i = completed_seqs_scores.index(max(completed_seqs_scores))
            seq = completed_seqs[i]
            alpha = torch.FloatTensor(completed_seqs_alpha[i])
            visualize_attention(image_path, seq, alpha, rev_wmap)