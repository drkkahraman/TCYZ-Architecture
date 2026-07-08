import argparse
import sys
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer
from tcyz.format import write_tcyz, TYPE_INT32, TYPE_FLOAT32, TYPE_STRING, read_tcyz
from tcyz.model import TCYZModel, ModelArgs

def main():
    parser = argparse.ArgumentParser(description="Train or fine-tune a TCYZ model on a text dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to plain text file for training")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="HF tokenizer name or path")
    parser.add_argument("--output", type=str, required=True, help="Path to save the trained .tcyz model")
    
    # Model config (used only if training from scratch)
    parser.add_argument("--dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of query attention heads")
    parser.add_argument("--n-kv-heads", type=int, default=None, help="Number of key/value attention heads")
    parser.add_argument("--hidden-dim", type=int, default=512, help="FeedForward hidden dimension")
    
    # Training hyperparams
    parser.add_argument("--fine-tune", type=str, default=None, help="Path to existing .tcyz model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence block size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu, cuda)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"Dataset path does not exist: {args.dataset}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loading tokenizer: {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    print(f"Loading dataset: {args.dataset}...")
    with open(args.dataset, "r", encoding="utf-8") as f:
        text = f.read()
        
    print("Tokenizing dataset...")
    enc = tokenizer(text, truncation=False, return_attention_mask=False)
    tokens = enc["input_ids"]
    print(f"Dataset has {len(tokens)} tokens.")
    
    if len(tokens) <= args.seq_len:
        print(f"Dataset size ({len(tokens)}) is too small for sequence length ({args.seq_len})", file=sys.stderr)
        sys.exit(1)

    model_args = None
    tensors = None
    metadata = {}
    
    if args.fine_tune:
        print(f"Loading existing model for fine-tuning from {args.fine_tune}...")
        meta, tensors = read_tcyz(args.fine_tune)
        model_args = ModelArgs(
            dim=meta["dim"],
            n_layers=meta["n_layers"],
            n_heads=meta["n_heads"],
            n_kv_heads=meta["n_kv_heads"],
            vocab_size=meta["vocab_size"],
            hidden_dim=meta["hidden_dim"],
            norm_eps=meta["norm_eps"],
            max_seq_len=meta["max_seq_len"],
            rope_theta=meta["rope_theta"]
        )
        metadata = meta
    else:
        print("Initializing model from scratch...")
        resolved_n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        model_args = ModelArgs(
            dim=args.dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_kv_heads=resolved_n_kv_heads,
            vocab_size=len(tokenizer),
            hidden_dim=args.hidden_dim,
            norm_eps=1e-5,
            max_seq_len=args.seq_len * 2,
            rope_theta=10000.0
        )
        metadata = {
            "dim": (TYPE_INT32, model_args.dim),
            "n_layers": (TYPE_INT32, model_args.n_layers),
            "n_heads": (TYPE_INT32, model_args.n_heads),
            "n_kv_heads": (TYPE_INT32, model_args.n_kv_heads),
            "vocab_size": (TYPE_INT32, model_args.vocab_size),
            "hidden_dim": (TYPE_INT32, model_args.hidden_dim),
            "norm_eps": (TYPE_FLOAT32, model_args.norm_eps),
            "max_seq_len": (TYPE_INT32, model_args.max_seq_len),
            "rope_theta": (TYPE_FLOAT32, model_args.rope_theta),
            "tokenizer_name": (TYPE_STRING, args.tokenizer)
        }
        
    model = TCYZModel(model_args)
    if tensors is not None:
        model.load_tensors(tensors, device=args.device)
        model.to(torch.float32)
    else:
        model.init_weights(device=args.device)
        
    model.train()
    
    # Simple data generator (non-overlapping blocks for standard and fast language model training)
    def get_batches():
        start_indices = torch.arange(0, len(tokens) - args.seq_len - 1, args.seq_len)
        indices = start_indices[torch.randperm(len(start_indices))]
        
        x_list = []
        y_list = []
        for idx in indices:
            start = idx.item()
            x_list.append(tokens[start : start + args.seq_len])
            y_list.append(tokens[start + 1 : start + args.seq_len + 1])
            
            if len(x_list) == args.batch_size:
                yield torch.tensor(x_list, device=args.device), torch.tensor(y_list, device=args.device)
                x_list = []
                y_list = []
                
    optimizer = AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    
    print(f"Starting training on {args.device}...")
    dtype = torch.float32 # use float32 for training stability
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        steps = 0
        
        for x, y in get_batches():
            optimizer.zero_grad()
            model.init_kv_cache(args.batch_size, model_args.max_seq_len, args.device, dtype)
            
            logits = model(x, 0, return_last_only=False)
            
            loss = loss_fn(logits.view(-1, model_args.vocab_size), y.view(-1))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            steps += 1
            
            if steps % 10 == 0:
                print(f"Epoch {epoch + 1}/{args.epochs} | Step {steps} | Loss: {loss.item():.4f}")
                
        if steps > 0:
            print(f"Epoch {epoch + 1} Complete | Average Loss: {epoch_loss / steps:.4f}")
            
    print(f"Saving trained model to {args.output}...")
    
    # Save the parameters back to .tcyz format (save as float16 to reduce disk footprint)
    trained_tensors = model.to_state_dict_tcyz()
    float16_tensors = {k: v.to(torch.float16) for k, v in trained_tensors.items()}
    
    # Update metadata if needed
    save_metadata = {}
    for k, v in metadata.items():
        if isinstance(v, tuple):
            save_metadata[k] = v
        else:
            # If read from fine-tune, it is already a direct value, re-wrap it
            if k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "hidden_dim", "max_seq_len"]:
                save_metadata[k] = (TYPE_INT32, int(v))
            elif k in ["norm_eps", "rope_theta"]:
                save_metadata[k] = (TYPE_FLOAT32, float(v))
            elif k in ["tokenizer_name"]:
                save_metadata[k] = (TYPE_STRING, str(v))
                
    write_tcyz(args.output, save_metadata, float16_tensors)
    print("Training and saving completed!")

if __name__ == "__main__":
    main()
