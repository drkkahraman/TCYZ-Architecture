# TCYZ Format & Inference Engine 1.0.1-alpha

https://tcyz.dorukk.dev/

A lightweight, high-performance, GGUF-like binary file format and inference architecture for Transformer-based LLMs. TCYZ allows memory-mapped loading, fast execution, and customizable inference options like streaming, temperature scaling, top-k, and top-p (nucleus) sampling.

## Key Features
- **GGUF-Like Binary Format**: Dedicated binary format (`.tcyz`) optimized for fast sequential loading and memory mapping (`mmap`).
- **Llama Architecture Support**: Full implementation of Rotary Position Embeddings (RoPE), RMSNorm, SwiGLU activation (MLP), and Grouped-Query Attention (GQA).
- **Interactive Generation**: High-performance streaming generation CLI with interactive chat and single-prompt modes.
- **Conversion Utilities**: Easy conversion from Hugging Face model formats to `.tcyz`.
- **Training from Scratch & Fine-Tuning**: Build a custom LLM and train/fine-tune it directly on raw text files, exporting the results directly into `.tcyz` format.

---

## File Format Specification (`.tcyz`)

A `.tcyz` file consists of the following components:

1. **Header**:
   - Magic Bytes: `TCYZ` (4 bytes)
   - Version: `uint32` (4 bytes, currently `1`)
   - Metadata Count: `uint64` (8 bytes)
   - Tensor Count: `uint64` (8 bytes)

2. **Metadata Key-Values**:
   For each metadata:
   - Key length (`uint32`)
   - Key string (`utf-8` encoded)
   - Value type (`uint32`):
     - `0`: INT32
     - `1`: FLOAT32
     - `2`: STRING
     - `3`: ARRAY_INT32
     - `4`: ARRAY_FLOAT32
   - Value content

3. **Tensor Metadata (Tensor Info List)**:
   For each tensor:
   - Name length (`uint32`)
   - Name string (`utf-8` encoded)
   - Dimension count (`uint32`)
   - Shape array (`uint32` array of size dimension count)
   - Data type (`uint32`):
     - `0`: FLOAT32
     - `1`: FLOAT16
     - `2`: BFLOAT16
     - `3`: INT8
   - Offset from the start of the tensor data block (`uint64`)
   - Size in bytes (`uint64`)

4. **Tensor Data Block**:
   - Aligned to 64 bytes for memory-mapped operations. Contains raw tensor binary data.

---

## Getting Started

### 1. Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Convert a Hugging Face Model to TCYZ
Convert any supported Hugging Face model (such as `Qwen/Qwen2.5-0.5B-Instruct` or `meta-llama/Llama-3.2-1B`) into the `.tcyz` format.

```bash
python convert_model.py --model Qwen/Qwen2.5-0.5B-Instruct --output qwen_0.5b.tcyz --dtype float16
```

### 3. Train or Fine-Tune a Model
You can train a model from scratch on a raw text file or fine-tune an existing `.tcyz` model.

**Pretraining from Scratch:**
To initialize a custom architecture and pretrain it on your text dataset:
```bash
python train_model.py \
    --dataset my_dataset.txt \
    --output trained_model.tcyz \
    --dim 256 \
    --n-layers 4 \
    --n-heads 8 \
    --hidden-dim 512 \
    --epochs 5 \
    --batch-size 8 \
    --seq-len 64 \
    --lr 5e-4
```

**Fine-Tuning an Existing Model:**
To fine-tune a model that was previously saved or converted:
```bash
python train_model.py \
    --dataset my_dataset.txt \
    --fine-tune trained_model.tcyz \
    --output fine_tuned_model.tcyz \
    --epochs 3 \
    --batch-size 4
```

### 4. Run Inference
You can run a single prompt or enter interactive chat mode.

**Single Prompt Mode:**
```bash
python run_inference.py --model trained_model.tcyz --prompt "Explain quantum computing in simple terms." --max-tokens 256
```

**Interactive Chat Mode:**
```bash
python run_inference.py --model trained_model.tcyz
```

### Options:
- `--temp`: Temperature scaling (default: `0.7`)
- `--top-k`: Keep only top k tokens (default: `40`)
- `--top-p`: Nucleus sampling threshold (default: `0.9`)
- `--device`: Execution device (`cpu` or `cuda`)
- `--max-tokens`: Max new tokens generated per turn (default: `128`)
