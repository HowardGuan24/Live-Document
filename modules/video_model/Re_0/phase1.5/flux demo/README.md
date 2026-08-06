# FLUX.2 Dev realism demos

Each case contains:

- `input-first-frame.png`: the exact first decoded frame from the Stage 1 MP4;
- `prompt.txt`: the case-specific structure-preserving realism prompt;
- `flux-output.png`: the generated FLUX.2 Dev result;
- `generation.json`: model, sampler, seed, dimensions, and ComfyUI provenance;
- `workflow-api.json`: the submitted ComfyUI API workflow.

Generation uses the local ComfyUI server and its bundled **Image Edit (Flux.2 Dev)** architecture: the source frame is VAE-encoded as a reference latent, while a same-aspect-ratio output latent is sampled with FLUX.2 Dev.

The full-resolution first frame is retained for audit. A 1536×864 copy is used for model conditioning and output to keep all dimensions divisible by the FLUX.2 VAE stride while preserving the 16:9 composition.

## Low-VRAM resumable run

Run `./run_lowvram.sh`. The launcher keeps the 12 GB text encoder and VAE on
CPU, uses tiled VAE encode/decode, enables DynamicVRAM with 10 GB of extra
headroom, reserves 6 GB of VRAM for sampling, uses PyTorch's memory-efficient
attention, and disables custom nodes. Both ComfyUI and generation logs are
written under `logs/` on persistent storage. Existing cases with both
`flux-output.png` and `generation.json` are skipped, so the command can safely
be run again after an interruption.

Specific cases can be selected as positional arguments. Set `FLUX_FORCE=1` to
regenerate an existing case, for example:

```bash
FLUX_FORCE=1 ./run_lowvram.sh mitosis
```
