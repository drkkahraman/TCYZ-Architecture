import torch
from transformers import AutoTokenizer
from .format import read_tcyz
from .model import TCYZModel, ModelArgs

class TCYZRunner:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        print(f"Loading TCYZ model from {model_path}...")
        self.metadata, self.tensors = read_tcyz(model_path)
        
        self.args = ModelArgs(
            dim=self.metadata['dim'],
            n_layers=self.metadata['n_layers'],
            n_heads=self.metadata['n_heads'],
            n_kv_heads=self.metadata['n_kv_heads'],
            vocab_size=self.metadata['vocab_size'],
            hidden_dim=self.metadata['hidden_dim'],
            norm_eps=self.metadata['norm_eps'],
            max_seq_len=self.metadata['max_seq_len'],
            rope_theta=self.metadata['rope_theta']
        )
        
        self.model = TCYZModel(self.args)
        self.model.load_tensors(self.tensors, device=self.device)
        self.model.eval()
        
        tokenizer_name = self.metadata.get('tokenizer_name', 'gpt2')
        print(f"Loading tokenizer {tokenizer_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens=100, temperature=1.0, top_k=50, top_p=0.9, stream_callback=None):
        tokens = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        seq_len = tokens.shape[1]
        
        self.model.init_kv_cache(1, self.args.max_seq_len, self.device, self.tensors[list(self.tensors.keys())[0]].dtype)
        
        # Prefill phase
        logits = self.model(tokens, 0)
        
        next_token = self._sample(logits, temperature, top_k, top_p)
        generated = [next_token.item()]
        
        if stream_callback:
            stream_callback(self.tokenizer.decode([generated[-1]]))
            
        curr_pos = seq_len
        
        for _ in range(max_new_tokens - 1):
            if generated[-1] == self.tokenizer.eos_token_id:
                break
                
            input_token = torch.tensor([[generated[-1]]], device=self.device)
            logits = self.model(input_token, curr_pos)
            
            next_token = self._sample(logits, temperature, top_k, top_p)
            generated.append(next_token.item())
            
            if stream_callback:
                stream_callback(self.tokenizer.decode([generated[-1]]))
                
            curr_pos += 1
            
        return self.tokenizer.decode(generated)

    def _sample(self, logits, temperature, top_k, top_p):
        if temperature == 0:
            return torch.argmax(logits, dim=-1)
            
        logits = logits / temperature
        
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
            
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('inf')
            
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(1)
