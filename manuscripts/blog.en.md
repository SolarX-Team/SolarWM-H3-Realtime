# SolarWM H3 Realtime Interaction

From faster inference to smooth, responsive interaction.

Keywords: Real-time interaction · Streaming generation

Real-time interaction depends on a world model responding promptly to the user's actions. With that goal in mind, we optimized the SolarWM H3 [1](#ref-1) inference pipeline in stages. Full-model compilation, attention backend selection, historical cache reuse, and CUDA Graph replay accelerate its MiniMax H3–based DiT [2](#ref-2). We then combine these improvements with streaming VAE decoding to reduce the time from an action to a generated frame.

We start with fixed-shape inference, improve the steady-state performance of each component, and bring those gains into an interactive deployment. Dynamic compilation and shape-keyed Graph caching let the service reuse prepared execution across supported scenes and prompt lengths. We also integrate the StreamRender-H3 decoder and weights [8](#ref-8) through our team's single-latent streaming adaptation [7](#ref-7), so the first frame responding to an action no longer has to wait for latents from the next generation chunk.

The static DiT path improves from **7.09 to 31.49 equivalent FPS, reaching 4.44× the original throughput**. Including VAE decoding, the complete serial backend improves from **4.92 to 24.38 steady-state FPS, reaching 4.95× the original throughput**. In one action-response test, the time until the corresponding first frame was ready at the backend fell from approximately **6.90 seconds to 0.58 seconds**, excluding network transfer and browser display. In the current web deployment, the measured median latency from keypress to the corresponding frame appearing on screen is **1.15 seconds**, including generation, transfer, and browser display.

We begin with a before-and-after video comparison of generation speed, followed by a timeline showing the backend response from action input to the first frame. We then explain how each stage reduces redundant computation and execution overhead, and how these improvements carry over to real-time interaction.

<!-- realtime-comparison -->

<!-- comparison-caption -->
Steady-state backend generation improves from **4.92 to 24.38 FPS, reaching 4.95× the original throughput**. Each side shows its corresponding 102-frame segment, played at a constant rate matching its measured steady-state serial DiT + VAE throughput.
<!-- /comparison-caption -->

<!-- timeline-note -->
The timeline compares action response before and after optimization, from the moment an action takes effect to the corresponding first frame becoming ready at the backend. The original MiniMax H3 decoder needs latents from the next chunk. Our single-latent version of StreamRender-H3 [8](#ref-8), developed in the streaming-decoder adaptation [7](#ref-7), can decode the current chunk's first frame without future inputs, providing visual feedback sooner. This measures first-frame response time, not the time to generate all 17 frames in a chunk, and excludes network transfer and browser display.
<!-- /timeline-note -->

## Accelerating DiT computation {#section-1}

To reduce action-to-frame latency, we start with the DiT, the most computationally expensive component. Conditioned on images, text, and actions, the DiT generates video latents as a stream. The VAE then decodes those latents into frames.

We first fix input shapes, including resolution and prompt length, and optimize compilation, attention, historical caches, and execution scheduling. This lets us establish the effect of each improvement before extending the optimized pipeline to an interactive service where users switch scenes and prompts.

Each complete generation chunk produces 5 latents, corresponding to 17 retained video frames. Its computation includes **four denoising forward passes and one historical key/value cache update (KV refresh)**. Refresh supplies clean context for subsequent generation, so its cost must be included in the per-chunk time.

For comparison, we convert the full DiT chunk time into an equivalent frame rate: `DiT FPS = 17 / (chunk time in milliseconds / 1000)`. The animation below reveals the optimizations stage by stage, showing how the DiT's equivalent generation rate rises from **7.09 to 31.49 FPS**. The table summarizes the time, frame rate, and main change at each stage.

<!-- stage-chart -->

| Optimization stage | Time per chunk | Equivalent DiT FPS | Main change |
| --- | ---: | ---: | --- |
| Native baseline | 2,396.49 ms | 7.09 | Original execution path |
| + Full-model Compile | 1,036.61 ms | 16.40 | Compile the full model with TorchInductor |
| + Attention Backend | 767.00 ms | 22.16 | Use SDPA for video queries and retain FlashAttention for the other groups |
| + History K/V Packing Reuse | 733.11 ms | 23.19 | Reuse preassembled historical K/V |
| + DiT Denoise Graph + KV Refresh Graph | 539.91 ms | 31.49 | Replay denoising and cache refresh using fixed buffers |

### Stage 1: Compile the full model {#section-2}

DiT execution costs extend beyond large matrix multiplications. Normalization, activations, element-wise operations, and tensor transformations are spread across 50 layers and repeated with every forward pass within a chunk. In eager execution, these operations are dispatched individually and produce intermediate results for subsequent operators to read. Each operation may be inexpensive, but their combined overhead is significant across the generation loop.

We use **`torch.compile` and TorchInductor** [3](#ref-3) to compile the complete 50-layer forward pass, with `fullgraph=True` requiring the entire forward path to enter the compilation graph. The compiler can then eliminate unnecessary intermediate reads and writes where operations can be fused, while reducing Python-level operator dispatch. “Full-model compilation” does not turn the entire model into one giant GPU kernel. It exposes the full forward computation to the compiler so that it can generate suitable code for each part.

Reuse also depends on input shapes. Historical context grows during startup and stabilizes only in steady state. Denoising and KV refresh follow different computational paths as well. We therefore prepare compiled entries for the relevant shapes and phases, and warm them up before measurement. The generation loop reuses compiled programs, while noise, camera conditions, and historical content still change on every call.

**Result: 1,036.61 ms per chunk, or 16.40 DiT FPS.** This stage reduces overhead across the full forward pass without changing the model architecture or generation step count. It also clarifies what compilation does and does not solve. The compiler can improve how operators are organized, but the suitability of an attention implementation for a given input still needs to be evaluated separately.

### Stage 2: Select the attention backend {#section-3}

We profile attention after compilation and compare backends within complete DiT inference. SolarWM H3 computes attention for conditioning, audio, and current-video queries in separate groups, with video queries reading a longer historical context. Measurements show that the original choice of FlashAttention for every group is not optimal for all of them. Switching the video-query group to SDPA makes complete inference faster.

We therefore switch **only the video-query group to PyTorch SDPA** [4](#ref-4), retaining **FlashAttention** [5](#ref-5) for the conditioning and audio groups. Runtime traces confirm that SDPA dispatches to **cuDNN Attention** in this test. The change preserves the original context scope, positional encoding, and camera conditioning. Only the attention implementation changes.

**Result: 767.00 ms per chunk, or 22.16 DiT FPS.** Compared with the compiled control measured at 1,051.29 ms per chunk in the same stage, this reduces latency by approximately **27%**.

### Stage 3: Reuse packed historical K/V {#section-4}

Profiling the compiled path reveals that, although the original KV cache stores historical keys and values, the individual history chunks are concatenated again before each attention call. That history does not change across the current chunk's four denoising passes. Repeating the concatenation creates unnecessary allocations and data copies.

We move history assembly to **cache refresh**. Each layer prepares the historical tensors needed by the next chunk in advance, allowing all four denoising passes to reuse them directly. The tensors are rebuilt when the history window changes, while prefix and current-chunk data are still assembled as needed. The optimization leaves cache contents and context scope unchanged, eliminating only repeated preparation of the same data.

**Result: 733.11 ms per chunk, or 23.19 DiT FPS.** The direct benefit comes from reducing repeated concatenation and its memory overhead. Preassembled history tensors also make it easier to populate **CUDA Graph's fixed input buffers**, preparing the data for Graph replay.

### Stage 4: DiT Denoise Graph + KV Refresh Graph {#section-5}

Compilation, attention backend selection, and historical K/V reuse make each forward pass substantially faster, but the CPU still launches GPU kernels one by one. Steady-state generation repeats the same computational structure, yet the work is submitted afresh on every forward pass. This overhead accumulates across each chunk's denoising passes and cache refresh, and faster GPU kernels alone cannot remove it.

We use **CUDA Graphs** [6](#ref-6) to record kernel launches and their dependencies, replacing individual submissions with replay of a captured execution path. Capture covers both **DiT denoising and KV refresh**. Refresh runs another forward pass on the denoised latents to build historical context for subsequent generation, making it an important acceleration target in its own right. All four denoising passes and the single KV refresh in a steady-state chunk run through their respective Graphs.

Once generation reaches steady state, the history window size and computational shapes remain stable, enabling Graph reuse. After warmup and capture, each replay updates the noise, timestep, camera conditions, and history without rebuilding the execution sequence. **What is reused is the execution path, not the input content.** New actions still reach the current chunk through its camera conditions.

**Result: 539.91 ms per chunk, or 31.49 DiT FPS.** Graph replay reduces per-kernel CPU submission and repeated scheduling, further shortening chunk generation without reducing denoising steps or skipping cache refresh.

## Accelerating VAE decoding {#section-6}

VAE latency has two distinct components: **how quickly decoding runs, and when all the inputs it needs become available**. The first is a question of computational efficiency. The second is determined by the temporal window. Both affect when the user sees a response to an action, but they require different optimizations.

To improve computational efficiency, we first compile the complete original H3 VAE decoding path with `fullgraph=True` and `dynamic=False`. We preserve its weights, attention backend, spatial tiling, 7-latent window, 5-latent stride, and 2-latent overlap. The fixed window gives the compiler stable shapes, reducing operator-by-operator execution overhead.

<!-- vae-chart -->

| Original VAE configuration | Time per 5-latent advance | Equivalent VAE FPS |
| --- | ---: | ---: |
| Uncompiled decoder | 821.87 ms | 20.68 |
| Compiled decoder | 450.84 ms | 37.71 |

As shown above, the standalone decoding test achieves **1.823× the original throughput and a 45.14% reduction in latency**. Input availability remains a separate constraint. The DiT produces 5 latents per chunk, but the original decoder requires a window of 7. It must wait for the next DiT chunk to supply the additional 2 latents before decoding the current frames. Making the VAE kernels faster cannot eliminate this wait for future inputs.

The base causal decoder and distilled weights used in the real-time deployment come from NVIDIA's **StreamRender-H3** [8](#ref-8). We integrate the **single-latent decoding adaptation** developed in our team's streaming-decoder work [7](#ref-7). It advances using available latents and historical state, without waiting for the next chunk to decode the current chunk's first frame. The DiT still produces 5 latents per chunk, while the VAE advances one latent at a time. This does not change the DiT's chunked generation scheme. This is the deployment shown in the opening comparison. Its weights and execution path differ from the compiled original VAE above, so their timings should not be treated as measurements of the same configuration.

## Real-time interaction results {#deployment}

Building on the DiT optimizations above, we enable **dynamic-shape compilation** for prompt length so that prompts of different lengths can share compiled code. Denoising and KV refresh Graphs are cached by input shape and reused on a cache hit, reducing repeated preparation when switching scenes. With the **StreamRender single-latent streaming decoder** [8](#ref-8)[7](#ref-7), the serial DiT + VAE backend reaches **24.38 steady-state FPS, or 4.95× the throughput of the original serial pipeline**. Single-latent decoding also removes the dependency on the next DiT chunk, bringing the backend readiness time for one action's first frame down to approximately **0.58 seconds**. In the current web deployment, the measured median latency from keypress to the corresponding frame appearing on screen is **1.15 seconds**, including generation, transfer, and browser display.

<!-- interaction-video-placeholder -->

## Conclusion {#section-9}

We optimize SolarWM H3's DiT inference in stages to support real-time interaction. Full-model compilation and attention backend selection improve computational efficiency. Reusing packed historical K/V and replaying CUDA Graphs reduce redundant work. Together, these changes increase the DiT's equivalent frame rate from **7.09 to 31.49 FPS, reaching 4.44× the original throughput**.

With the **StreamRender single-latent streaming decoder** [8](#ref-8)[7](#ref-7), the serial DiT + VAE backend improves from **4.92 to 24.38 steady-state FPS, reaching 4.95× the original throughput**. Single-latent decoding also removes the first frame's dependency on the next generation chunk. In the action-response test recorded here, the time from an action being accepted by the backend to its first frame becoming ready falls from approximately **6.90 seconds to 0.58 seconds**.

For real-time deployment, dynamic-shape compilation for prompt length and shape-keyed caching of denoising and KV refresh Graphs reduce repeated preparation when switching inputs. In the current web deployment, the measured median latency from keypress to the corresponding frame appearing on screen is **1.15 seconds**, including generation, transfer, and browser display. From faster computation and streaming decoding to reusable execution, these improvements reduce the wait between an action and its visual response, making world models better suited to real-time interaction.

## References {#section-11}

<!-- references -->

## Citation {#citation}

<!-- citation -->
