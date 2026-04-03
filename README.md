# LiGGUF

**LiGGUF** is (probably) the first **tiny, dependency-free LLM inference engine** with **direct GGUF support**.
No conversions, no helper scripts, no Python detours.

The most recent "FAST" update introduced a load of new features and improvements. Check them out:

## New: faster inference using AVX2 (on x86) and NEON (on ARM)

The new `fast/` engine is no longer just a compact "proof that it runs".
It has architecture-specific hot paths for **AVX2/FMA** on x86 and **NEON** on ARM, while still keeping the codebase small enough to read in one sitting.

So the idea stays the same, but the execution path is much closer to something you would actually want to embed into a serious low-latency system.

## New: 1-bit quantization support

The FAST engine now supports both **Q1_0** and **Q1_G** (aka `Q1_0_G128`) alongside the classic **Q8_0** and the full **Qn_K** family.

That matters because 1-bit inference is no longer just a research-paper talking point.
Models like **Bonsai-8B** push the idea into "real model you can actually run" territory, which is a much bigger deal than the old BitNet (which was actually ternary) ever was.
Once practical 1-bit models exist as something more than slides and papers, a tiny direct-loader like LiGGUF becomes a very interesting tool and a foundation for bigger projects.

## New: support for Qwen3 models

LiGGUF FAST now supports **Qwen3-class GGUF models** as well, including the architecture-specific details they need at inference time.

That is useful because Qwen3 has become one of the most widely used open model families, not just for general chat, but also for coding and agentic work.
So this is not niche compatibility for one weird checkpoint, but support for a model family people actually use.

## New: support for BERT models and RAG

LiGGUF FAST can also load a **BERT GGUF** model and use it as a tiny local embedding engine for chat memory retrieval.

The mechanism is intentionally simple.
It parses a previous chat log, embeds user turns, keeps a small in-memory vector database, and injects the most relevant past records back into the prompt before the model replies.
So yes, LiGGUF now has a lightweight, self-contained form of **RAG-style memory**, without turning into a giant framework.

## New: SARA distributed example

LiGGUF now also includes an experimental distributed inference example at `cpp/ligguf_distrib.cpp`

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

## New: chat mode

The FAST engine now has an actual **chat mode**, not just single-shot prompt completion.

It can keep a running conversation, save and reload model state, store chat logs, inject user/AI names, and stop on configurable end-of-turn markers.
So the whole thing starts feeling much less like a toy launcher and much more like a tiny terminal-first chat runtime.

## New: hotpath visualization with SDL2

If you build the SDL2-enabled variant, LiGGUF can show you a live visualization of the hottest activation paths while the model runs.

That is not just eye candy.
It is a rare chance to literally watch signal flow move through embeddings, attention output, feed-forward output, and final logits token by token.
If your idea of fun includes seeing a neural network "think" in real time, this feature is absurdly satisfying.

And this only library dependency is still totally optional.

---

## What it is

LiGGUF is now really **two closely related things**:

- tiny, self-contained **single-file reference implementations** in C and C++ flavors
- a still-small but much more capable **FAST engine** in `fast/`

The single-file editions are there because sometimes the best way to understand a transformer is to read the whole thing front to back with no indirection and no scaffolding.

The FAST engine keeps the same minimalist spirit, but splits the code into a few focused modules so it can grow into something more practical:
more quantization formats, more architectures, chat, retrieval, visualization, and CPU-specific acceleration.

Now available in **four practical forms**:

- a clear and educational **C++ edition**
- a smaller and faster **pure C edition**
- an experimental **distributed C++ example** in `cpp/ligguf_distrib.cpp`, demonstrating **SARA**
- the modular **FAST engine** in `fast/`, intended as the serious path forward

---

## Highlights

- **Still tiny:** the original C and C++ engines remain compact enough to study line by line on a single sheet of A4.
- **No dependency stack:** the core path is plain C/C++, and the FAST engine only adds **optional** SDL2 for visualization.
- **Direct GGUF parsing:** opens `.gguf` files directly via `mmap`, with no conversion or preprocessing step.
- **Wide quant support:** **Q8_0**, **Q1_0**, **Q1_G**, and **Q2_K/Q3_K/Q4_K/Q5_K/Q6_K**.
- **More than one model family:** classic LLaMA-style models, **Qwen3**, and even **BERT** for embeddings.
- **Readable:** the code is still small enough that every subsystem can be inspected without archaeology.
- **Functional:** runs full transformer inference with attention, RMSNorm, RoPE, SwiGLU, KV cache, and Top-P/Top-K/greedy sampling.
- **Interactive:** chat mode, saved state, prompt files, chat logs, and optional memory retrieval.
- **Visual:** the SDL2 build can render live hotpath traces while the network runs.
- **Distributed example included:** `cpp/ligguf_distrib.cpp` shows how the same minimalist style can be extended toward multi-CPU inference with **SARA**.

---

## Why it exists

LiGGUF started as a debugging experiment and evolved into a statement:
you don't need a huge overengineered blob with millions of LOCs just to use a LLaMA model.
You need only about 780 lines of C (or 900 lines of C++) to understand the essentials.

Then it kept growing in the only acceptable direction: not toward bloat, but toward capability.
The result is a package with both a tiny educational core and a much faster, more practical engine that is still lean enough to understand and small enough to control.

Not "small for the sake of small", but small enough to trust, small enough to port, and **small enough to replace a shocking amount of heavier software** when all you actually need is local inference.

**It can also be embedded!** You can run it on almost anything, including bare-metal targets, provided the board has enough RAM ;)

For a 7B model (32 layers) and a 4K context window, you'll need only 512.34 MiB. This is within reach for many contemporary devkits.

The distributed example exists for the same reason: to show you how to tie together a bunch of cheap CPUs to make inference faster.

---

## Building it

It doesn't need a heavy external build system of a particular version. It relies on standard tools found in every *NIX system.

So it can't be simpler:

```
make
```

It will build everything, including **FAST** build with **AVX2/FMA** on x86 or **NEON** on ARM by default.

---

## Running it

### Single-file reference implementations

Grab any LLaMA-style **Q8_0** GGUF model, for example from Hugging Face, then run one of the minimal core versions:

```
./ligguf-c model.gguf 64 "Hello there"
```

or, if you want to feed raw token IDs:

```
./ligguf-cpp model.gguf 32 tokens 1 15043 29871
```

LiGGUF will tokenize, run inference, and print tokens directly.

### Distributed example

The distributed example in `cpp/ligguf_distrib.cpp` is intended for experimentation with **SARA-style multi-CPU inference**.

Conceptually, it is meant to be launched as cooperating workers that:

- load the same GGUF model
- take slightly different roles or shard assignments
- exchange activation fragments across a network link or local transport
- reconstruct the layer result with minimal synchronization overhead

And it's super easy to run:

```
# worker
./ligguf-cpp-distrib -m Mistral-7B-Instruct-v0.3-Q8_0.gguf -W 1/2 -M 19095

# master
./ligguf-cpp-distrib -m Mistral-7B-Instruct-v0.3-Q8_0.gguf -w 10.0.0.2:19095 -n 64 Hi
```

### FAST version

The FAST engine is the one you want if you need the newer quant formats, Qwen3 support, chat mode, retrieval, or visualization.

Supported options:

- `-m <model.gguf>`: model to load
- `-n <tokens>`: number of tokens to generate
- `-s <seed>`: RNG seed
- `-c <n_context>`: override context length up to model maximum
- `-p <prompt>`: prompt passed on the command line
- `-f <file>`: load prompt from file
- `-l <chat_log_file>`: load and later update a chat log file
- `-S <state_file>`: save model state after prompt prefill
- `-L <state_file>`: load a previously saved model state
- `-E <end_of_turn_marker>`: stop generation when this marker appears
- `-A <AI_name>`: AI speaker name for chat mode
- `-U <User_name>`: user speaker name for chat mode
- `-K <TopK>`: Top-K sampling limit
- `-T <TopP>`: Top-P sampling threshold
- `-M <temperature>`: sampling temperature
- `-B <bert.gguf>`: BERT model used for retrieval memory
- `-G`: greedy sampling
- `-C`: chat mode
- `-V`: enable visualizer

Single-shot generation:

```
./ligguf-fast -m model.gguf -n 64 -p "Explain rotary embeddings in one paragraph."
```

Generation from a prompt file:

```
./ligguf-fast -m model.gguf -f prompt.txt -n 128 -K 40 -T 0.9 -M 0.7
```

Chat mode:

```
./ligguf-fast -m model.gguf -C -A Assistant -U User -E "User:" -n 256
```

Chat mode with memory retrieval from a previous log:

```
./ligguf-fast -m model.gguf -C -l chat.log -B bert.gguf -A Assistant -U User -E $'\n' -n 256
```

Prompt prefill plus state save/load:

```
./ligguf-fast -m model.gguf -f system_prompt.txt -S state.bin
./ligguf-fast -m model.gguf -L state.bin -n 128
```

Visualization build:

```
./ligguf-fast-sdl -m model.gguf -p "Trace this token stream." -n 64 -V
```

The FAST engine supports **Q8_0**, **Q1_0**, **Q1_G**, and the **Qn_K** family from **Q2_K** through **Q6_K**.
It also understands both classic **LLaMA**-style checkpoints and **Qwen3** models, while a separate **BERT** GGUF can be attached for retrieval memory.

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

### FAST engine

- AVX2/FMA hot paths on x86 and NEON hot paths on ARM
- Direct support for **Q8_0**, **Q1_0**, **Q1_G**, **Q2_K**, **Q3_K**, **Q4_K**, **Q5_K**, and **Q6_K**
- LLaMA-style architecture support
- **Qwen3** architecture support, including its extra attention normalization path
- **BERT** GGUF loading for embeddings
- Top-K, Top-P, temperature, and greedy sampling
- Prompt-file input and saved-state prefill/load workflow
- Terminal chat mode
- Lightweight vector-memory retrieval over prior chat logs
- Optional SDL2 hotpath visualization

### Additional SARA example

- A **SARA**-style distributed inference example in `cpp/ligguf_distrib.cpp`
- Multi-CPU sharding of selected inference work
- Activation exchange and reduction back into the residual path
- A compact reference point for low-latency, single-stream inference experiments on edge

---

## Not included

- Millions of lines of code bloat
- A giant framework with 42 abstraction layers
- An enormous CMake file
- Mandatory dependency on every build system and every library under the Sun
- F'ng telemetry

---

## Philosophy

> "Simplicity is not lack of complexity - it's control over it."

LiGGUF exists to be read, understood, and tinkered with.
It’s the codebase you can open in any text editor (or print on a single A4 sheet), read through, and actually *get it*.

If you feel like taking the next step, it also contains a much more serious path, with acceleration, support for multiple models, chat, retrieval, visualization, and even distributed inference.

Basically, if you want to build an AI system from scratch, or with a minimalist foundation, LiGGUF is the best option to start with.

---

## Credits

Created by **Dmitry 'sciloaf' Solovyev** aka **MatrixS_Master**, in 2025-2026.

Thanks to **llama.cpp** and **llama2.c** projects for inspiration (and GGUF format :).

## License

MIT
