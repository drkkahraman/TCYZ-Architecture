import argparse
import sys
import torch
from tcyz.runner import TCYZRunner

def main():
    parser = argparse.ArgumentParser(description="Run inference using a TCYZ model")
    parser.add_argument("--model", type=str, required=True, help="Path to .tcyz model file")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for text generation (if not provided, enters interactive chat)")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on (cpu, cuda)")
    parser.add_argument("--chat", action="store_true", help="Format the input as chatbot conversation (Kullanıcı: / Yapay Zeka:)")
    
    args = parser.parse_args()
    
    try:
        runner = TCYZRunner(args.model, device=args.device)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)
        
    if args.prompt:
        prompt = f"Kullanıcı: {args.prompt}\nYapay Zeka:" if args.chat else args.prompt
        print(f"\n--- Prompt ---\n{prompt}\n\n--- Output ---")
        
        def stream_cb(token):
            print(token, end="", flush=True)
            
        runner.generate(
            prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            stream_callback=stream_cb
        )
        print()
    else:
        print("\nEntering interactive chat mode. Type 'exit' or 'quit' to end.\n")
        while True:
            try:
                user_input = input("User > ")
                if user_input.strip().lower() in ["exit", "quit"]:
                    break
                if not user_input.strip():
                    continue
                
                print("AI > ", end="", flush=True)
                
                prompt = f"Kullanıcı: {user_input}\nYapay Zeka:" if args.chat else user_input
                
                def stream_cb(token):
                    print(token, end="", flush=True)
                    
                runner.generate(
                    prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temp,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    stream_callback=stream_cb
                )
                print("\n")
            except KeyboardInterrupt:
                print("\nExiting...")
                break

if __name__ == "__main__":
    main()
