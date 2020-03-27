import torch
import torch.nn as nn
import torchvision
from commons import *

class Encoder(nn.Module):
    def __init__(self, encoded_image_size=14, fine_tune=True):
        super(Encoder, self).__init__()
        self.encoded_image_size = encoded_image_size
        
        resnet = torchvision.models.resnet101(pretrained=True)
        all_modules = list(resnet.children())[:-2]
        
        self.resnet = nn.Sequential(*all_modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.encoded_image_size, self.encoded_image_size))
        
        self.fine_tune(fine_tune)
    
    def fine_tune(self, fine_tune):
        for p in self.resnet.parameters():
            p.requires_grad = False
        
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune
    
    def forward(self, images):
        out = self.resnet(images)
        out = self.adaptive_pool(out)
        return out.permute(0, 2, 3, 1)


class SelfAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(SelfAttention, self).__init__()
        
        self.enc2attn = nn.Linear(encoder_dim, attention_dim)
        self.dec2attn = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
    
    def forward(self, encoder_out, decoder_hid):
        att1 = self.enc2attn(encoder_out)
        att2 = self.dec2attn(decoder_hid)
        attn = self.full_att(torch.relu(att1+att2.unsqueeze(1))).squeeze(2)
        alpha = torch.softmax(attn, dim=1)
        
        weighted_enc = (encoder_out*alpha.unsqueeze(2)).sum(dim=1)
        
        return weighted_enc, alpha

class Decoder(nn.Module):
    def __init__(self, attention_dim, embedding_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super(Decoder, self).__init__()
        
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        
        self.attention = SelfAttention(self.encoder_dim, self.decoder_dim, self.attention_dim)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.decode_step = nn.LSTMCell(self.embedding_dim+self.encoder_dim, self.decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)
        
        self.init_weights()
    
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0.0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
    
    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)
    
    def fine_tune_embeddings(self, fine_tune):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune
    
    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        
        return h,c
    
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        pixel_count = encoder_out.size(1)
        
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        
        embeddings = self.embedding(encoded_captions)
        
        h, c = self.init_hidden_state(encoder_out)
        
        decode_lengths = (caption_lengths-1).tolist()
        max_len = max(decode_lengths)
        
        predictions = torch.zeros(batch_size, max_len, vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), pixel_count).to(device)
        
        for t in range(max_len):
            batch_size_t = sum([l>t for l in decode_lengths])
            attn_enc, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            
            gate = torch.sigmoid(self.f_beta(h[:batch_size_t]))
            attn_enc = gate * attn_enc # elementwise multiplication
            h, c = self.decode_step(torch.cat([embeddings[:batch_size_t,t,:], attn_enc], dim=1), (h[:batch_size_t], c[:batch_size_t]))
            
            pred = self.fc(self.dropout(h))
            
            predictions[:batch_size_t, t, :] = pred
            alphas[:batch_size_t, t, :] = alpha
        
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
    
if __name__ == "__main__":
    pass
    Encoder()
    SelfAttention(10,10,10)
    Decoder(10,10,20,30)