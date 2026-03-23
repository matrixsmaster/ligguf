# Ligguf

**Ligguf** is (probably) the first **tiny, dependency-free LLaMA inference engine** with **direct GGUF support**.
No conversions, no helper scripts, no Python detours.

It is also unique in that it exists in **pure C** and **C++** forms side by side, with the same minimal spirit throughout.

Just one file, one compiler of your choice, and a model straight from Hugging Face.

---

## New: SARA distributed example

Ligguf now also includes an experimental distributed inference example at `cpp/ligguf_distrib.cpp`

This file demonstrates **[SARA: Sharded Activation Reduction Architecture](https://blog.syntheva.no/2026/03/sara-sharded-activation-reduction-architecture)**.

SARA is a lightweight, robotics-friendly approach to distributed LLM inference:

- shard selected transformer work across multiple CPUs or nodes
- exchange compact intermediate activations
- reduce partial results back into the residual stream

The motivation is simple.
In robotics, there is often only **one active stream of thought** to accelerate, but several CPUs may be available.
That makes **single-stream latency** more interesting than aggregate datacenter throughput.

This example is meant as a compact research playground, not a giant serving framework.
It keeps the ligguf philosophy intact: minimal, direct, readable, and close to the metal.

---

## What it is

Ligguf is a fully self-contained program that loads and runs quantized LLaMA-family **GGUF** models *directly from disk*.
It is a stand-alone, minimal, end-to-end implementation of the whole model pipeline in under 900 lines for the core C++ version, or roughly 780 lines for the C version.

Now available in **three practical forms**:

- a clear and educational **C++ edition**
- an even smaller and much faster **pure C edition**
- an experimental **distributed C++ example** in `cpp/ligguf_distrib.cpp`, demonstrating **SARA**

---

## Highlights

- **Tiny:** under **900** lines for the core C++ version, and about **780** for the C version, comments included.
- **No dependencies:** standard library only.
- **Direct GGUF parsing:** opens `.gguf` files directly via `mmap`, with no conversion or preprocessing step.
- **Quantized:** supports **Q8_0** out of the box, using the same layout as `llama.cpp`.
- **Readable:** every step is spelled out.
- **Functional:** runs full transformer inference with attention, RMSNorm, RoPE, SwiGLU, and greedy sampling.
- **Educational:** an ideal codebase to learn *how* a transformer actually runs, byte by byte.
- **Now with a distributed example:** `cpp/ligguf_distrib.cpp` shows how the same minimalist style can be extended toward multi-CPU inference with **SARA**.

---

## Why it exists

Because "small and clear" is still the best way to understand "big and complex."

Ligguf started as a debugging experiment and evolved into a statement:
you don't need ten repositories and a build farm to understand how a LLaMA thinks.
You need only about 780 lines of C (or 900 lines of C++).

**It can also be embedded!** You can run it on almost anything, including bare-metal targets, provided the board has enough RAM ;)

For 7B model (32 layers) and 4K context window, you'll need only 512.34 GiB. This is within reach for many contemporary devkits.

The distributed example exists for the same reason - to show you how to tie together a bunch of cheap CPUs to make inference faster.

---

## Building it

Can't be simpler:

```
make
```

The `Makefile` also includes a **debug** target:

```
make debug
```

Use it only if you enjoy scrolling through thousands of lines of floating-point enlightenment.

---

## Running it

Grab any LLaMA-style **Q8_0** GGUF model, for example from Hugging Face, then run one of the core versions:

```
./ligguf-c model.gguf 64 "Hello there"
```

or, if you want to feed raw token IDs:

```
./ligguf-cpp model.gguf 32 tokens 1 15043 29871
```

Ligguf will tokenize, run inference, and print tokens directly.

### Distributed example

The distributed example in `cpp/ligguf_distrib.cpp` is intended for experimentation with **SARA-style multi-CPU inference**.

Conceptually, it is meant to be launched as cooperating workers that:

- load the same GGUF model
- take slightly different roles or shard assignments
- exchange activation fragments across a network link or local transport
- reconstruct the layer result with minimal synchronization overhead

Or simply speaking:

```
# worker
./ligguf-distrib -m Mistral-7B-Instruct-v0.3-Q8_0.gguf -W 1/2 -M 19095

# master
./ligguf-distrib -m Mistral-7B-Instruct-v0.3-Q8_0.gguf -w 10.0.0.2:19095 -n 64 Hi
```

---

## Features implemented

### Core ligguf engine

- Full **GGUF v3** parser (metadata + tensors)
- Tokenizer and score-based merge logic
- FP16/FP32 conversion
- Q8_0 quantization and dequantization
- RMSNorm
- RoPE
- Multi-head attention with **GQA**
- Simple per-layer **KV cache**
- **SwiGLU** feed-forward network
- Greedy sampler
- Multithreading

### Additional experimental example

- A **SARA**-style distributed inference example in `cpp/ligguf_distrib.cpp`
- Multi-CPU sharding of selected inference work
- Activation exchange and reduction back into the residual path
- A compact reference point for low-latency, single-stream inference experiments on edge

Everything else is left as an exercise for the inspired.

---

## Not (yet) included

- Sanity
- A giant framework with twelve build systems and forty-seven abstractions

---

## Philosophy

> "Simplicity is not lack of complexity - it's control over it."

Ligguf exists to be read, understood, and tinkered with.
It’s the codebase you can open in a text editor (or print on a single A4 sheet), read through, and actually *get it*.

If you feel like taking the next step, it also contains a compact path into distributed inference research without abandoning that philosophy.

---

## Credits

Created by **Dmitry 'sciloaf' Solovyev** aka **MatrixS_Master**, in 2025-2026.

Thanks to **llama.cpp** and **llama2.c** projects for inspiration (and data types).

## License

MIT
