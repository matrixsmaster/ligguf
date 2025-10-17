# Ligguf

**Ligguf** is (probably) the first **tiny, dependency-free LLaMA inference engine** with **direct GGUF support** —
no conversions, no helper scripts, no Python detours.
Just one file, one compiler, and a model from Hugging Face.

Yes, it really works.

---

## What it is

Ligguf is a fully self-contained C++ program that loads and runs quantized LLaMA-family **GGUF** models *directly from disk*.
It is a stand-alone, minimal, end-to-end implementation — the whole model pipeline in under 800 lines.

---

## Highlights

- **Tiny:** under **800 lines**, including comments.
- **No dependencies:** standard library only.
- **Direct GGUF parsing:** opens `.gguf` files directly via mmap — no conversions, no preprocessing.
- **Quantized:** supports **Q8_0** out of the box (same layout as llama.cpp).
- **Readable:** every step is spelled out — no templates, no macros, no mysterious helpers.
- **Functional:** runs full transformer inference with attention, RMSNorm, RoPE, SwiGLU, and greedy sampling.
- **Educational:** an ideal codebase to learn *how* a transformer actually runs, byte by byte.

---

## Why it exists

Because “small and clear” is still the best way to understand “big and complex.”

Ligguf started as a debugging experiment and evolved into a statement:
you don’t need ten repositories and a build farm to understand how a LLaMA thinks.
You need about 800 lines of C++ and an unhealthy amount of curiosity.

---

## Building it

You have two options, depending on your level of courage:

1. **For the fearless:**
```
g++ -O3 ligguf.cpp -o ligguf
```

2. **For those who don't dare touching bare g++ without safety gloves:**
```
make
```

The `Makefile` also includes a **debug** target:
```
make debug
```
which builds `ligguf_debug` — the version that prints *everything*.
Use only if you enjoy scrolling through thousands of lines of floating-point enlightenment.

---

## Running it

Grab any LLaMA-style **Q8_0** GGUF model (for example from Hugging Face), then:

```
./ligguf model.gguf "Hello"
```

or, if you want to feed raw token IDs:

```
./ligguf model.gguf tokens 1 15043 29871
```

Ligguf will tokenize, run inference, and print tokens directly.
It doesn’t stream or batch — it just works, one token at a time.

---

## Features implemented

- Full **GGUF v3** parser (metadata + tensors)
- Tokenizer and score-based merge logic
- FP16/FP32 conversion
- Q8_0 quantization and dequantization
- RMSNorm
- RoPE (rotary position embedding)
- Multi-head attention with **GQA**
- Simple per-layer **KV cache**
- **SwiGLU** feed-forward network
- Greedy sampler

Everything else is left as an exercise for the inspired.

---

## Not (yet) included

- Speed
- Multithreading
- Temperature or top-k sampling
- Sanity

---

## Philosophy

> “Simplicity isn’t lack of complexity — it’s control over it.”

Ligguf exists to be read, understood, and tinkered with.
It’s the codebase you can open in a text editor (or print on a single A4 sheet), read through, and actually *get it*.
Every number has a reason. Every line has a purpose.

---

## Credits

Created by **Dmitry 'sciloaf' Solovyev** aka **MatrixS_Master**, in 2025, for fun!

Thanks to **llama.cpp** and **llama2.c** projects for inspiration (and data types).

## License

License? Do you need it? Fine! MIT then :)
