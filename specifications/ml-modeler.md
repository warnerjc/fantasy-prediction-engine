---
persona: ml-modeler
status: resolved — CPU-only LightGBM confirmed sufficient, GPU available but not planned for use
---

# ML Modeler — Local System Questions

These confirm the bootstrap plan's "LightGBM trains in seconds-to-minutes, no GPU needed"
assumption actually holds on your hardware, and whether training happens on the same machine
as everything else.

1. Roughly how many CPU cores and how much RAM does the machine that would run training have?
   (Confirms the "seconds-to-minutes" assumption holds for walk-forward validation across
   potentially 20+ years of history, and whether the nflverse pull window should be trimmed to
   keep training fast.)

   *Answer:* i7-12700KF (3.6 GHz, 12 cores / 20 logical processors), 64 GB DDR4 RAM. GPU: an
   RTX "3700" as reported (8 GB dedicated VRAM, ~47.8 GB shared) — that's not a standard Nvidia
   SKU name, likely a 3070/3070 Ti; the exact model doesn't matter much below since we're not
   planning to use it (see Q3).

   **Assessment:** comfortably more than the bootstrap plan needs. LightGBM/XGBoost on weekly
   player-week rows (even pulling full nflverse history back to 1999, across all positions) is
   a dataset in the hundreds-of-thousands-of-rows range at most — this hardware trains that in
   seconds, walk-forward-validates across 20+ held-out seasons without needing to trim the pull
   window at all. No compute-driven reason to scope down the data pull.

2. Will model training run on the same Windows machine as the data pipeline and application, or
   is there another machine (or willingness to use a cloud notebook for heavier experimentation)
   you'd reserve for it?

   *Answer:* Same machine (which is also where the pipeline/app run, per `data-engineer.md` and
   `AGENTS.md`). Has some Google Colab experience as a fallback if ever needed, but doesn't
   expect to need it given the hardware headroom above.

3. Any interest in GPU-accelerated boosting (LightGBM GPU build), or is CPU-only fine given the
   expected scale? (Mainly confirming we shouldn't complicate the environment setup for no
   benefit.)

   *Answer:* Open to it if needed ("have a GPU to support it to some extent"), but not required.

   **Recommendation: stay CPU-only.** GPU-accelerated gradient boosting pays off at dataset
   sizes far larger than this project's (millions-to-billions of rows) — at this scale it adds
   real setup cost (CUDA drivers, `nvidia-container-toolkit` for Docker GPU passthrough) for no
   measurable speed benefit, and complicates the container image for nothing. Keep the GPU as
   an unused fallback; only revisit if a specific future bottleneck (e.g., a local LLM for the
   v2 injury-news parsing idea, if that's ever built to run locally rather than via an API)
   actually needs it.
