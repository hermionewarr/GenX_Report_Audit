import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class GenXConfig:
    block_size: int = 512 #1024 #256 <- see if this is enough for the reports.
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 8 #16 #8 #12
    n_head: int =  8 #16. 8
    n_embd: int = 512 #256  #1024 #512 #512 #768 #800
    im_embd: int = 512
    dropout: float = 0.0 #0.25 #0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    concat: bool = True #True # whether to concatenate the image features to the token embeddings
    
class GenX(nn.Module):

    def __init__(self, config, no_im_tokens):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.bos_token_id = 50256
        self.eos_token_id = 50256
        self.no_im_tokens = no_im_tokens

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            #wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, no_im_tokens) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.rope = RotaryPositionalEmbedding(config.n_embd, config.block_size)
        self.im_embed = nn.Linear(config.im_embd, config.n_embd)
        """ self.im_embed = nn.sequential( #if im embeds = nembd #try this llava 1.5 onwards found it better than linear
            MLP(config), 
            MLP(config)
        ) """
        self.im_embed2 = nn.Linear(config.im_embd, no_im_tokens*config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        print(f"no image tokens gpt {no_im_tokens}")

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        #if non_embedding:
        #    n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, 
                idx, 
                im_feats, 
                attn_mask = None,
                targets = None, 
                ):
        device = im_feats.device
        b, t = idx.size()
        no_im_tokens = self.no_im_tokens
        if attn_mask is not None:
            #MASK 
            # get a boolean copy of the attention_mask and invert it
            mask_to_ignore_padding_tokens_for_loss_computation = ~(attn_mask.to(torch.bool))
            attn_mask = attn_mask.view(b, -1)
            attn_mask = attn_mask[:, None, None, :]
            # since we have 1 additional column in the attention weights due to the additional concatenated matrix
            # of the image hidden states, we have to shift the attention mask "one to the right" and add a column of ones
            # to the left such that the attention weights corresponding to the image are not masked out
            attention_mask_size = attn_mask.size()
            if self.config.concat:
                ones_column = torch.ones(attention_mask_size[:-1] + (1,), dtype=torch.int64, device=device)  # shape [batch_size, 1, 1, 1]
                attn_mask = torch.cat((ones_column, attn_mask), dim=-1)
                if no_im_tokens >1:
                    for i in range(no_im_tokens-1):
                        attn_mask = torch.cat((ones_column, attn_mask), dim=-1)
                       
        
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        tok_emb = self.transformer.wte(idx) # token embeddings
        if self.config.concat:
            pos = torch.arange(0, t+no_im_tokens, dtype=torch.long, device=device) # shape (t)
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device)

        #here concat the image features with the tok emdeddings
        if self.config.concat:
            if no_im_tokens == 1:
                im_feats = self.im_embed(im_feats)
                im_feats = im_feats[:, None, :] 
            if no_im_tokens >1:
                im_feats = self.im_embed2(im_feats).view(b, no_im_tokens, -1) 
            tok_emb = torch.cat((im_feats, tok_emb), dim=1) 
        
        pos_emb = self.rope(pos) #rotary positional embeddings
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x, attn_mask) 

        x = self.transformer.ln_f(x)
        
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x) 
            if self.config.concat:
                logits = logits[:, no_im_tokens:, :]  #[drop the first token, the im feats] #[2, 512, 50304]
            shift_logits = logits[:,:-1,:].contiguous()
            labels = targets 
            # set padding tokens to -100, such that they are ignored and don't count towards the loss
            labels[mask_to_ignore_padding_tokens_for_loss_computation] = -100
            shift_labels = labels[:,1:].contiguous()
            # flatten the tokens
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

            return shift_logits, loss
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            #logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim ????
            logits =self.lm_head(x)
            loss = None

            return logits, loss
    
    @torch.no_grad()
    def generate(self, 
                 im_feats,
                 max_new_tokens, 
                 #no_im_tokens, 
                 top_k=5, 
                 idx=None):
        # start with the bos_token_id for all image features in the batch. 
        no_im_tokens = self.no_im_tokens
        batch_size = im_feats.size(0)
        if idx is None:
            idx = torch.full(size=(batch_size, 1), fill_value=self.bos_token_id, dtype=torch.int64, device=device)
        attention_mask = torch.ones(size=(batch_size, idx.shape[-1]), dtype=torch.int64, device=device)

        for _ in range(max_new_tokens):
            logits, _ = self(idx, im_feats, attention_mask)
            logits = logits[:, -1, :]
            idx_next = torch.argmax(logits, dim=-1)
            idx_next = idx_next.unsqueeze(-1)
            idx = torch.cat((idx, idx_next), dim=-1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
            torch.cuda.empty_cache()

            for i in range(idx.size(0)): 
                row = idx[i]
                indices = (row == self.eos_token_id).nonzero()
                if len(indices) > 1:
                    first_occurrence_index = indices[1].item() #1 rather than 0 as start of report has bos token
                    idx[i, first_occurrence_index+1:] = self.eos_token_id
        return idx
    
class CausalSelfAttention(nn.Module):

    def __init__(self, config, no_im_tokens):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.concat = config.concat
        self.config = config
        #self.no_im_tokens = no_im_tokens
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
        if self.concat:
            #self.flash = False #to have more control over the attention mask
            #(to make sure the mask does not get updated during backprop)
            tril = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size) 
            self.register_buffer("bias", tril)
        
    def forward(self, x, attention_mask):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        
        T_q = q.size(1)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T_q, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_q, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T_q, hs) x (B, nh, hs, T) -> (B, nh, T_q, T)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            attention_mask = (1.0 - attention_mask) * -100000.0
            att = att.masked_fill(self.bias[:,:,:T_q,:T] == 0, float('-inf')) #+ attention_mask
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T_q, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T_q, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class Block(nn.Module):
    def __init__(self, config, no_im_tokens):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, no_im_tokens)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_positions):
        super(RotaryPositionalEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.max_positions = max_positions
        self.inv_freq = 1. / (10000 ** (torch.arange(0, embed_dim, 2).float() / embed_dim))

    def forward(self, positions):
        """
        Args:
            positions (torch.Tensor): 1-D tensor containing the sequence positions.

        Returns:
            torch.Tensor: Rotary positional embeddings for the given positions.
        """
        sinusoids = torch.einsum("i,d->id", positions.float(), self.inv_freq.to(device))
        embeddings = torch.cat([sinusoids.sin(), sinusoids.cos()], dim=-1)
        return embeddings[:, :self.embed_dim]