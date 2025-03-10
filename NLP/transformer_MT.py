# !pip install transformers
# !pip install sentencepiece # MarianTokenizer 불러올 때 필요
# !pip install sacremoses # MarianMTModel 에서 불러올 때 warning 뜨는 것 방지
# !pip install einops # Einstein operations => rearrange 함수를 위해

#%%
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pandas as pd
from tqdm import tqdm
import math, random
from einops import rearrange

# for random seed
random_seed = 0
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(random_seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#%%
### **허깅페이스 Pretrained translation Model**

from transformers import MarianMTModel, MarianTokenizer # MT: Machine Translation

# Load the tokenizer & model
tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ko-en')
model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-ko-en') # MT: Machine Translation
# input_embedding = model.get_input_embeddings() # 일부 사이즈 큰 놈들은 데려와서 나머지만 학습시키려는 시도
# input_embedding.weight.requires_grad = False # freeze
# fc_out = model.get_output_embeddings()
# fc_out.weight.requires_grad = False

eos_idx = tokenizer.eos_token_id
pad_idx = tokenizer.pad_token_id
print("eos_idx = ", eos_idx)
print("pad_idx = ", pad_idx)

#%%
# tokenizer 써보기 (_로 띄어쓰기를 나타낸다! 즉, _가 없으면 이어진 한 단어임을 나타냄 subword tokenizing)
# 토크나이저에 대해 참고 자료: https://ratsgo.github.io/nlpbook/docs/preprocess/bpe/
print(tokenizer.tokenize("Hi, I'm Hyuk. ...        a   a?"))
print(tokenizer.tokenize("a/b 1+2+3 2:1 a>b"))
print(tokenizer.tokenize("pretrained restart"))
print(tokenizer.tokenize("chatGPT"))
print(tokenizer.tokenize("The example is very good in our lecture")) # 띄어쓰기도 tokenize 할 때가 있다.
print(tokenizer.tokenize("한글은 어떻게 할까?"))
print(tokenizer.tokenize("확실히 띄어쓰기 기준으로 토크나이징을 하는 것 같진 않다."))
print(tokenizer.tokenize("여러분들 차례!"))

#%%
# print(tokenizer.get_vocab()) # 단어사전 확인
vocab_size = tokenizer.vocab_size
print(vocab_size)

print(tokenizer.encode('지능', add_special_tokens=False)) # string to index
print(tokenizer.encode('<pad>', add_special_tokens=False)) # <pad>는 65000
print(tokenizer.encode('</s>', add_special_tokens=False)) # <sos> or <eos>는 0
print(tokenizer.encode('He', add_special_tokens=False)) # add_special_tokens=False 는 <eos> 자동 붙여주는 것을 방지
print(tokenizer.encode('he', add_special_tokens=False)) # 대소문자 다른 단어로 인식

#%%

print(tokenizer.tokenize('문장을 넣으면 토크나이즈해서 숫자로 바꾼다'))
print(tokenizer.encode('문장을 넣으면 토크나이즈해서 숫자로 바꾼다', add_special_tokens=False))
print(tokenizer.decode([204]))
print(tokenizer.decode([206]))
print(tokenizer.decode([210]))
print(tokenizer.decode(list(range(15)) + [65000,65001,65002,65003]))

#%%
# 사전 학습된 모델로 번역해보기 (생각보다 성능 좋네)
input_text = "헐! 대박 쩐다!"

input_tokens = tokenizer.encode(input_text, return_tensors="pt")
translated_tokens = model.generate(input_tokens, max_new_tokens=100)
translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

print("입력:", input_text)
print("AI의 번역:", translated_text)

#%% 

## **하이퍼 파라미터 설정**

BATCH_SIZE = 64 # 논문에선 2.5만 token이 한 batch에 담기게 했다고 함.
LAMBDA = 0 # l2-Regularization를 위한 hyperparam. # 저장된 모델
EPOCH = 15 # 저장된 모델
# max_len = 512 # model.model.encoder.embed_positions 를 보면 512로 했음을 알 수 있다.
max_len = 100 # 너무 긴거 같아서 자름 (GPU 부담도 많이 덜어짐)
criterion = nn.CrossEntropyLoss(ignore_index = pad_idx) # pad token 이 출력 나와야하는 시점의 loss는 무시 (즉, label이 <pad> 일 때는 무시) # 저장된 모델
# criterion = nn.CrossEntropyLoss(ignore_index = pad_idx, label_smoothing = 0.1) # 막상 해보니 성능 안나옴 <- 데이터가 많아야 할 듯

scheduler_name = 'Noam'
# scheduler_name = 'Cos'
#### Noam ####
# warmup_steps = 4000 # 이건 논문에서 제시한 값 (총 10만 step의 4%)
warmup_steps = 1000 # 데이터 수 * EPOCH / BS = 총 step 수 인것 고려 # 저장된 모델
LR_scale = 0.5 # Noam scheduler에 peak LR 값 조절을 위해 곱해질 녀석 # 저장된 모델
#### Cos ####
LR_init = 5e-4
T0 = 1500 # 첫 주기
T_mult = 2 # 배 만큼 주기가 길어짐 (1보다 큰 정수여야 함)
#############

new_model_train = False
hyuk_model_use = True # 여러분만의 모델 만들어서 사용하고 싶다면 False로

if hyuk_model_use:
    # !gdown https://drive.google.com/uc?id=1bjbeWgqlVKJ9gzDcL1hfD9cIItQy9_N2 -O Transformer_small.pt
    # !gdown https://drive.google.com/uc?id=1M0yYP2umxlwaAbk_iq5G_Z5y3qLu9Wet -O Transformer_small_history.pt

    save_model_path = './result_models/Transformer_small.pt'
    save_history_path = './result_models/Transformer_small_history.pt'
else:
    save_model_path = './result_models/Transformer_small2.pt'
    save_history_path = './result_models/Transformer_small2_history.pt'


# 논문에 나오는 base 모델 (train loss를 많이 줄이려면 많은 Epoch이 요구됨, 또, test 성능도 좋으려면 더 많은 데이터 요구)
# n_layers = 6
# d_model = 512
# d_ff = 2048
# n_heads = 8
# drop_p = 0.1

# 좀 사이즈 줄인 모델 (훈련된 input_embedding, fc_out 사용하면 사용 불가)
n_layers = 3        # encoder, decoder layer 수
d_model = 256        # embedding size
d_ff = 512         # feed forward hidden size   
n_heads = 8       # attention head 수
drop_p = 0.1        # dropout 확률
# %%
from torch.utils.data import DataLoader, random_split, Dataset
import pandas as pd 

# 사용 데이터 
data = pd.read_excel(r'..\dt\한국어-영어 번역(병렬) 말뭉치\2_대화체.xlsx')

# 커스텀 데이터셋 만들기
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data.loc[idx, '원문'], self.data.loc[idx, '번역문']

custom_DS = CustomDataset(data)

train_DS, val_DS, test_DS = random_split(custom_DS, [97000, 2000, 1000])


#%%
BATCH_SIZE = 16
train_DL = DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=False)
val_DL = DataLoader(val_DS, batch_size=BATCH_SIZE, shuffle=False)
test_DL = DataLoader(test_DS, batch_size=BATCH_SIZE, shuffle=False)

print(len(train_DS))
print(len(val_DS))
print(len(test_DS))

#%%
# test_DS 확인
i = 5
src_text, trg_text = test_DS[i]

print(f"인덱스: {test_DS.indices[i]}")  # 엑셀 파일에서 idx+2번째 문장에 들어있음을 확인할 수 있다 (엑셀은 1번째 시작하고 1열엔 "원문", "번역문" 이런 열 정보가 써있음)
print(f"원문: {src_text}")
print(f"번역문: {trg_text}")

# %%
# train_DL 테스트
for src_texts, trg_texts in train_DL:

    # print(src_texts)
    # print(trg_texts)
    # print(len(src_texts))
    # print(len(trg_texts))
    
    # # 여러 문장에 대해서는 tokenizer.encode() 가 아닌 그냥 tokenizer()
    src = tokenizer(src_texts, padding=True, truncation=True, max_length = max_len, return_tensors='pt', add_special_tokens = False).input_ids # pt: pytorch tensor로 변환
    
    # add_special_tokens = True (default)면 마지막에 <eos> 를 붙임
    # truncation = True: max_len 보다 길면 끊고 <eos> 집어넣어버림
    
    # 타겟 문장에 대해서는 <sos> 를 붙여줌
    trg_texts = ['</s> ' + s for s in trg_texts]
    
    # <sos>가 토크나이저에 따로 없길래 </s> 를 <sos> 로도 사용
    trg = tokenizer(trg_texts, padding=True, truncation=True, max_length = max_len, return_tensors='pt', add_special_tokens = True).input_ids
    
    print(src.shape)
    print(trg.shape)
    print(src[:2])
    print(trg[:2])
    
    print(src.shape)   # (batch_size, seq_len)  64개의 문장중에서 가장 긴 문장의 길이수 
    print(trg.shape)
    
    print(trg[:,-1]) # 가장 마지막 단어를 보니 어떤 문장은 <eos> 로 끝이 났고 나머지는 <pad> 로 끝이 났다는 걸 볼 수 있음
    print(tokenizer.decode(trg[trg[:,-1]==eos_idx,:][0])) # 가장 긴 문장 중 첫 번째 문장 관찰
    
    print(trg[5,:-1]) # 디코더 입력
    print(trg[5,1:]) # 디코더 출력
    # 그런데 [:,:-1] 로 주면 패딩된 문장은 eos도 넣는 셈 아닌가? 맞다! 하지만 괜찮다. 어차피 출력으로 pad token이 기다리고 있으니.. (loss에서 ignore됨)

    break
# %%
# 내가 쓸 train data 에 대해서 MarianMTmodel이 잘 번역하는지 확인
src_text, trg_tex = test_DS[5]
print(f"입력: {src_text}")
print(f"정답: {trg_text}")

# add_special_tokens = False 해보면 뭔가 이상하게 번역함 (학습 때 source에도 <eos>를 넣었단 증거?)
src = tokenizer.encode(src_text, return_tensors='pt', add_special_tokens = True)

translated_tokens = model.generate(src, max_new_tokens=max_len)
translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=False)

print(f"AI의 번역: {translated_text}") # 디코더 첫 입력으로 <pad> 토큰을 넣었음. (<pad>를 <sos>로 사용)

#%%

src_text = '나는 지금 딥러닝 공부를 하고 있는데 자연어 처리가 너무 어렵게 느껴져요'

src = tokenizer.encode(src_text, return_tensors='pt', add_special_tokens = True)

# add_special_tokens = False 해보면 뭔가 이상하게 번역함 (학습 때 source에도 <eos>를 넣었단 증거?)  : 학습할 때 <eos>를 넣어서 학습했는데 추론할때 넣지 않아서 그럼럼
translated_tokens = model.generate(src, max_new_tokens=max_len)
translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=False)

print(f"입력: {src_text}")
print(f"AI의 번역: {translated_text}") # 디코더 첫 입력으로 <pad> 토큰을 넣었음. (<pad>를 <sos>로 사용)

#%%

## **모델 구현**

## **Positional Encoding**


## **Multi Head Attention**
class MHA(nn.Module):
    def __init__(self, d_model, n_heads):   # d_model: embedding 차원, n_heads: head 수
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)
        
        self.scale = torch.sqrt(torch.tensor(d_model / n_heads))    # d_model / n_heads 가 정수이면 상관없지만 아니면 // 연산자를 사용해야함
        # self.scale = torch.tensor(d_model / n_heads) ** 0.5
        
    def forward(self, Q, K, V, mask = None):
        
        # (batch_size, seq_len, d_model) = (한번에 들어가는 문장의 개수, 단어의 개수, 임베딩 차원)
        Q = self.fc_q(Q)  
        K = self.fc_k(K)
        V = self.fc_v(V)
        
        # split heads    
        Q = rearrange(Q, 'b s (h d) -> b h s d', h = self.n_heads)   # (batch_size, seq_len, d_model) -> (batch_size, n_heads, seq_len, head_dim)
        K = rearrange(K, 'b s (h d) -> b h s d', h = self.n_heads)    # b, s, d에서 d를 (h, d) = (h x d) 라고 할거고 h = n_heads 라고 지정할거야 
        V = rearrange(V, 'b s (h d) -> b h s d', h = self.n_heads)
        
        # attention score : Q * K^T / sqrt(d_k)   # transpose(-2, -1) : -2번째 차원과 -1번째 차원을 바꾼다(s, d -> d, s) s: sequence length, d: d_model
        attention_score = Q @ K.transpose(-2, -1) / self.scale   
        
        # masked self attention : 미래시점의 값에 극단적인 음수값을 주어 softmax를 거친 후 0에 가깝게 만들어줌(decoder), 
        # encoder는 패딩 토큰을 제외하고 나머지 토큰들끼리만 attention을 하기 위해 패딩을 마스크함
        # 극단적인 음수값을 주어 Encoder에서는 패딩 토큰무시, Decoder에서는 미래 시점의 토큰 무시 + 패딩 토큰 무시시
        if mask is not None:
            attention_score[mask] = -1e10
            # attention_score = attention_score.masked_fill(mask == 0, -1e10)  # mask가 0인 부분을 -1e10으로 채워라
        
        # Attention weight : softmax(attention) 
        attention_weights = torch.softmax(attention_score, dim = -1)
        
        # attention value : attention_weight * V   attention weight와 V의 weighted sum
        attention = attention_weights @ V
        
        # Concatenate heads   num_heads * head_dim = d_model
        # num_heads와 head_dim을 다시 d_model로 합쳐주기 위해 1, 2의 위치를 바꾸고 view를 하여 합쳐줌 
        x = rearrange(attention, 'b h s d -> b s (h d)')  # 개 해 단 차 -> 개 단 차
        
        x = self.fc_o(x)
        
        return x, attention_weights

        # rearrange를 사용하지 않을때 
        # batch_size = Q.size(0)
        
        # Q = self.fc_q(Q)
        # K = self.fc_k(K)
        # V = self.fc_v(V)
        
        # Q = Q.reshape(batch_size, -1, self.n_heads, self.d_model // self.n_heads).permute(0, 2, 1, 3)   # permute(0, 2, 1, 3) == transpose(2, 1)
        # K = K.reshape(batch_size, -1, self.n_heads, self.d_model // self.n_heads).permute(0, 2, 1, 3)
        # V = V.reshape(batch_size, -1, self.n_heads, self.d_model // self.n_heads).permute(0, 2, 1, 3)
        
        # attention_score = Q @ K.transpose(-2, -1) / self.scale
        
        # if mask is not None:
        #     attention_score[mask] = -1e10
            
        # attention_weights = torch.softmax(attention_score, dim = -1)
        
        # attention = attention_weights @ V
        
        # x = attention.permute(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        
        ### view, reshape 사용
        # x = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # x = attention.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        
        
        # x = self.fc_o(x)
        
        # return x, attention_weights
        
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop_p):
        super().__init__()
        
        self.linear = nn.Sequential(nn.Linear(d_model, d_ff),
                                    nn.ReLU(),
                                    nn.Dropout(drop_p),
                                    nn.Linear(d_ff, d_model))
        
    def forward(self, x):
        x = self.linear(x)
        return x 
        
# %%
        
## **Encoder Layer**

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, drop_p):
        super().__init__()
        
        self.atten = MHA(d_model, n_heads)
        self.atten_norm = nn.LayerNorm(d_model)
        
        self.FF = FeedForward(d_model, d_ff, drop_p)
        self.FF_norm = nn.LayerNorm(d_model)
        
        self.drop = nn.Dropout(drop_p)
        
    def forward(self, x, enc_mask):

        """
        Args:
            x: 입력 텐서 (batch_size, seq_len, d_model)
            enc_mask: 패딩 토큰 마스크 (batch_size, seq_len, seq_len)
        
        Returns:
            x: Encoder Layer의 출력 (batch_size, seq_len, d_model)
        """
               
        # Multi-head attention   # 인코더의 마스크 왜 있냐 : 패딩 토큰을 제외하고 나머지 토큰들끼리만 attention을 하기 위해 패딩을 마스크함 
        residual_mha, atten_enc = self.atten(x, x, x, enc_mask)  # atten_enc는 시각화를 위한 것이므로 출력하지 않아도 무방
        
        # Add & Norm(Residual Connection & Layer Normalization)
        
        # dropout 
        residual_mha = self.drop(residual_mha)
        
        # Residual Connection & Layer Normalization
        x = self.atten_norm(x + residual_mha)
        
        # Feed Forward
        residual_FF = self.FF(x)
        
        # dropout
        residual_FF = self.drop(residual_FF)
        
        # Add & Norm(Residual Connection & Layer Normalization)
        x = self.FF_norm(x + residual_FF)
        
        return x, atten_enc
        
        
#%%

## Encoder ## 
class Encoder(nn.Module):
    def __init__(self, input_embedding, max_len, n_layers, d_model, d_ff, n_heads, drop_p):
        super().__init__()
        
        self.scale = torch.sqrt(torch.tensor(d_model))
        self.input_embedding = input_embedding        
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        self.dropout = nn.Dropout(drop_p)
        
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, n_heads, drop_p) for _ in range(n_layers)])
    
    def forward(self, src, mask, atten_map_save = False):

        # 시퀀스 길이 만큼 위치를 나타내는 pos_embedding을 만들어줌
        pos = torch.arange(src.shape[1]).expand_as(src).to(DEVICE)
        
        # input embedding + pos embedding을 해서 위치정보를 가진 임베딩을 만들어줌
        x = self.scale * self.input_embedding(src) + self.pos_embedding(pos)
        x = self.dropout(x)
        
        atten_encs = torch.tensor([]).to(DEVICE)
        
        for layer in self.layers:
            x, atten_enc = layer(x, mask)
            if atten_map_save:
                atten_encs = torch.cat([atten_encs, atten_enc[0].unsqueeze(0)], dim = 0)
                
                
        return x, atten_encs
    

#%%
## **Decoder Layer**

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, drop_p):
        super().__init__()
        
        # Maksed Self Attetnion
        self.masked_attention = MHA(d_model, n_heads)
        self.masked_attention_norm = nn.LayerNorm(d_model)
        
        # Encoder-Deocder Attention
        self.en_de_attention = MHA(d_model, n_heads)
        self.en_de_attention_norm = nn.LayerNorm(d_model)
        
        # FeedForward
        self.FF = FeedForward(d_model, d_ff, drop_p)
        self.FF_norm = nn.LayerNorm(d_model)
        
        # dropout
        self.dropout = nn.Dropout(drop_p)
        
    def forward(self, x, enc_out, dec_mask, enc_dec_mask):
        
        # Masked Self Attention
        residual_mha, _ = self.masked_attention(x, x, x, dec_mask)  # (Q, K, V, mask)
        
        # Add & Norm(Residual Connection & Layer Normalization)
        
        # dropout
        residual_mha = self.dropout(residual_mha)
        
        # Residual Connection & Layer Normalization
        x = self.masked_attention_norm(x + residual_mha)
        
        # Encoder - Decoder Attention(Q는 디코더로부터 K,V는 인코더로부터!!)
        residual_atten, _ = self.en_de_attention(x, enc_out, enc_out, enc_dec_mask) 
        
        # Add & Norm(Residual Connection & Layer Normalization)
        
        # dropout
        residual_atten = self.dropout(residual_atten)
        
        # Residual Connection, Layer Norm 
        x = self.en_de_attention_norm(x + residual_atten)
        
        # Feed Forward
        residual_FF = self.FF(x)
        residual_FF = self.dropout(residual_FF)
        
        # Add & Norm
        x = self.FF_norm(x + residual_FF)
        
        return x

#%%  
## **Decoder**   
class Decoder(nn.Module):
    def __init__(self, input_embedding, max_len, n_layers, d_model, d_ff, n_heads, drop_p):
        super().__init__()
        
        # Output Embedding
        self.input_embedding = input_embedding
        
        # Positional Embedding
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        # dropout
        self.dropout = nn.Dropout(drop_p)
        
        # Decoder Layers
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, drop_p) for _ in range(n_layers)])
        
        # FC Layer
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, trg, enc_out, dec_mask, enc_dec_mask, atten_map_save = False):
        
        # Positional Embedding
        pos = torch.arange(trg.shape[1]).expand_as(trg)   # (batch_size, seq_len)
        x = self.input_embedding(trg) + self.pos_embedding(pos)  # (batch_size, seq_len, d_model)
        
        # dropout 
        x = self.dropout(x)
        
        atten_decs = torch.tensor([]).to(DEVICE)
        atten_enc_decs = torch.tensor([]).to(DEVICE)
        for layer in self.layers:
            x, atten_dec, atten_enc_dec = layer(x, enc_out, dec_mask, enc_dec_mask)
            if atten_map_save is True:
                atten_decs = torch.cat([atten_decs , atten_dec[0].unsqueeze(0)], dim=0) # 층헤단단 ㅋ
                atten_enc_decs = torch.cat([atten_enc_decs , atten_enc_dec[0].unsqueeze(0)], dim=0) # 층헤단단 ㅋ

        x = self.fc_out(x)

        return x, atten_decs, atten_enc_decs

#%%
## **Transformer**
class Transformer(nn.Module):
    def __init__(self, vocab_size, max_len, n_layers, d_model, d_ff, n_heads, drop_p):
        super().__init__
        
        # Input Embedding
        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = Encoder(self.input_embedding, max_len, n_layers, d_model, d_ff, n_heads, drop_p)
        self.decoder = Decoder(self.input_embedding, max_len, n_layers, d_model, d_ff, n_heads, drop_p)
        
        self.n_heads = n_heads

        # weight initialization
        
        # for m in self.modules():
        #     if hasattr(m,'weight') and m.weight.dim() > 1: # layer norm에 대해선 initial 안하겠다는 뜻
        #         nn.init.kaiming_uniform_(m.weight) # Kaiming의 분산은 2/Nin

        for m in self.modules():
            if hasattr(m,'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight) # xavier의 분산은 2/(Nin+Nout) 즉, 분산이 더 작다. => 그래서 sigmoid/tanh에 적합한 것! (vanishing gradient 막기 위해)
                
                
    # Encoder mask(입력 시퀀스의 패딩 토큰을 마스킹)
    def make_enc_mask(self, src): # src: (batch_size, seq_len) (개 단)
        
        # src의 각 값이 pad_idx와 같은지 비교하여 패딩 위치를 True/False로 나타내는 텐서를 생성
        enc_mask = (src == pad_idx).unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, seq_len)
        enc_mask = enc_mask.expand(src.shape[0], self.n_heads, src.shape[1], src.shape[1]) # (batch_size, n_heads, seq_len, seq_len)

        """ src pad mask (문장 마다 다르게 생김. 이건 한 문장에 대한 pad 행렬)
        F F T T
        F F T T
        F F T T
        F F T T
        """
               
        return enc_mask
    
    # Decoder masked self attention mask(현재 시점 이후의 단어들을 마스킹)
    def make_dec_mask(self, trg):
        
        # trg의 각 값이 pad_idx와 같은지 비교하여 패딩 위치를 True/False로 나타내는 텐서를 생성
        trg_pad_mask = (trg == pad_idx).unsqueeze(1).unsqueeze(2)
        trg_pad_mask = trg_pad_mask.expand(trg.shape[0], self.n_heads, trg.shape[1], trg.shape[1])

        """ trg pad mask
        F F F T T
        F F F T T
        F F F T T
        F F F T T
        F F F T T
        """
        
        # 대각선을 기준으로 위쪽을 True로 채움
        # torch.tril() : 텐서의 **하삼각 행렬(lower triangular matrix)**만 남기고, 나머지 위치는 0으로 설정, 상삼각 행렬은 0으로 하삼각 행렬은 1로로
        # 1로 채워진 행렬을 만들고, 하삼각 행렬만 1로 변경하고 ==0을 하여 1은 False, 0은 True로 변경
        trg_future_mask = torch.tril(torch.ones(trg.shape[1], self.n_heads, trg.shape[1], trg.shape[1])) ==0
        trg_future_mask = trg_future_mask.to(DEVICE)
        
        """ trg future mask
        F T T T T
        F F T T T
        F F F T T
        F F F F T
        F F F F F
        """

        # 두 마스크를 합침(두 마스크 중 하나라도 True이면 True)  패딩위치가 True 이거나 미래 시점의 단어가 True이면 True
        dec_mask = trg_pad_mask | trg_future_mask
        """ decoder mask
        F T T T T
        F F T T T
        F F F T T
        F F F T T
        F F F T T
        """
        return dec_mask

    # Encoder-Decoder 패딩 mask 
    def make_enc_dec_mask(self, src, trg):

        enc_dec_mask = (src == pad_idx).unsqueeze(1).unsqueeze(2) # 개11단
        enc_dec_mask = enc_dec_mask.expand(trg.shape[0], self.n_heads, trg.shape[1], src.shape[1]) # 개헤단단
        """ src pad mask
        F F T T
        F F T T
        F F T T
        F F T T
        F F T T
        """
        return enc_dec_mask
    
    def forward(self, src, trg):
        
        # Encoder Mask
        enc_mask = self.make_enc_mask(src)
        
        # Decoder Mask
        dec_mask = self.make_dec_mask(trg)
        
        # Encoder-Decoder Mask
        enc_dec_mask = self.make_enc_dec_mask(src, trg)
        
        # Encoder
        enc_out, atten_encs = self.encoder(src, enc_mask)
        
        # Decoder
        dec_out, atten_decs, atten_enc_decs = self.decoder(trg, enc_out, dec_mask, enc_dec_mask)
        
        return dec_out, atten_encs, atten_decs, atten_enc_decs
    
    
# %%
