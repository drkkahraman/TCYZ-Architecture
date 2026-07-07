import argparse
import sys
from tcyz.convert import convert_hf_to_tcyz

def main():
    parser = argparse.ArgumentParser(description="Convert Hugging Face model to TCYZ (.tcyz) format")
    parser.add_argument("--model", type=str, required=True, help="HF model hub name or local path")
    parser.add_argument("--output", type=str, required=True, help="Path to output .tcyz file")
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "float32"], default="float16", help="Target tensor data type")
    
    args = parser.parse_args()
    
    try:
        convert_hf_to_tcyz(args.model, args.output, args.dtype)
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
