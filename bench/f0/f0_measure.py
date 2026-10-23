"""F0 measurement: one "description -> song" run on Apple Silicon, MLX path.

Prints a JSON line per phase with wall time, process footprint, MLX and MPS
memory, and the MLX flags, so silent fallbacks to torch are visible.
Run under `/usr/bin/time -l` to also get the kernel's peak footprint.
"""
import json
import os
import subprocess
import sys
import time

PROJECT = os.path.expanduser("~/acestep-f0/ACE-Step-1.5")
CKPT = os.path.join(PROJECT, "checkpoints")
OUT = os.path.expanduser("~/acestep-f0/out")
QUERY = os.environ.get(
    "F0_QUERY", "a melancholic italian folk ballad with acoustic guitar and a warm male voice"
)
DURATION = float(os.environ.get("F0_DURATION", "30"))
SEED = 42
T0 = time.time()


def footprint_mb() -> float:
    """Return this process's physical footprint in MB via `footprint`, or -1."""
    try:
        out = subprocess.run(
            ["footprint", "-p", str(os.getpid())], capture_output=True, text=True, timeout=30
        ).stdout
        for line in out.splitlines():
            if "phys_footprint:" in line:
                val, unit = line.split("phys_footprint:")[1].split()[:2]
                return float(val) * {"KB": 1 / 1024, "MB": 1, "GB": 1024}[unit]
    except Exception:  # noqa: BLE001
        pass
    return -1.0


def swap_mb() -> float:
    """Return system swap used in MB."""
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(out.split("used = ")[1].split("M")[0].replace(",", "."))


def mem_stats() -> dict:
    """Collect MLX and MPS allocator stats when available."""
    stats = {}
    try:
        import mlx.core as mx

        stats["mlx_active_mb"] = round(mx.get_active_memory() / 2**20)
        stats["mlx_peak_mb"] = round(mx.get_peak_memory() / 2**20)
        stats["mlx_cache_mb"] = round(mx.get_cache_memory() / 2**20)
    except Exception as exc:  # noqa: BLE001
        stats["mlx_err"] = str(exc)[:80]
    try:
        import torch

        stats["mps_driver_mb"] = round(torch.mps.driver_allocated_memory() / 2**20)
        stats["mps_current_mb"] = round(torch.mps.current_allocated_memory() / 2**20)
    except Exception as exc:  # noqa: BLE001
        stats["mps_err"] = str(exc)[:80]
    return stats


def report(phase: str, started: float, **extra) -> None:
    """Print one JSON line for a finished phase."""
    rec = {
        "phase": phase,
        "phase_s": round(time.time() - started, 2),
        "elapsed_s": round(time.time() - T0, 2),
        "footprint_mb": round(footprint_mb()),
        "swap_used_mb": round(swap_mb()),
        **mem_stats(),
        **extra,
    }
    print("F0 " + json.dumps(rec), flush=True)


def slim_after_init(dit) -> None:
    """F1 experiment: drop the torch decoder weights and keep the MLX decoder in bf16.

    The MLX decoder already holds its own copy, so the torch one is redundant on
    the MLX path. Parameters are replaced by empty tensors to keep the module
    structure for code that only inspects it.
    """
    import gc

    import mlx.core as mx
    import torch

    if not getattr(dit, "use_mlx_dit", False):
        raise RuntimeError("MLX DiT not active, refusing to drop the torch decoder")
    with torch.no_grad():
        for p in dit.model.decoder.parameters():
            p.data = torch.empty(0, dtype=p.dtype, device=p.device)
    gc.collect()
    torch.mps.empty_cache()
    if os.environ.get("F0_MLX_BF16") == "1":
        dit.mlx_decoder.set_dtype(mx.bfloat16)
        mx.eval(dit.mlx_decoder.parameters())
        mx.clear_cache()
    torch_bf16 = os.environ.get("F0_TORCH_BF16") == "1"
    if torch_bf16 or os.environ.get("F0_COMPACT") == "1":
        # Round-trip through CPU so the MPS allocator can release the heaps
        # that the emptied decoder left fragmented.
        for name in ("model", "text_encoder"):
            getattr(dit, name).to("cpu")
        gc.collect()
        torch.mps.empty_cache()
        target = torch.bfloat16 if torch_bf16 else dit.dtype
        for name in ("model", "text_encoder"):
            setattr(dit, name, getattr(dit, name).to(target).to(dit.device))
        if torch_bf16:
            dit.dtype = torch.bfloat16
        gc.collect()
        torch.mps.empty_cache()


def fixed_params(path: str, GenerationParams):
    """Build GenerationParams that replay saved inputs with the LM fully skipped."""
    with open(path) as fh:
        d = json.load(fh)
    return GenerationParams(
        caption=d["caption"], lyrics=d["lyrics"], bpm=d.get("bpm"), duration=d["duration"],
        keyscale=d.get("keyscale", ""), timesignature=d.get("timesignature", ""),
        vocal_language=d.get("vocal_language", "unknown"), audio_codes=d["audio_codes"],
        thinking=False, use_cot_metas=False, use_cot_caption=False, use_cot_language=False,
    )


def save_fixed(path: str, audio: dict) -> None:
    """Save the conditioning actually used by a generation so it can be replayed."""
    keys = ("caption", "lyrics", "bpm", "duration", "keyscale", "timesignature",
            "vocal_language", "audio_codes", "seed")
    prm = audio.get("params", {})
    with open(path, "w") as fh:
        json.dump({k: prm.get(k) for k in keys}, fh, ensure_ascii=False, indent=1)


def main() -> int:
    """Run init, create_sample and generate_music once, reporting each phase."""
    sys.path.insert(0, PROJECT)
    os.makedirs(OUT, exist_ok=True)
    report("start", T0)

    t = time.time()
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, create_sample, generate_music
    from acestep.llm_inference import LLMHandler
    report("imports", t)

    t = time.time()
    dit = AceStepHandler()
    status, ok = dit.initialize_service(PROJECT, "acestep-v15-turbo", device="auto")
    report(
        "init_dit", t, ok=ok,
        device=str(getattr(dit, "device", "?")), dtype=str(getattr(dit, "dtype", "?")),
        use_mlx_dit=getattr(dit, "use_mlx_dit", None), use_mlx_vae=getattr(dit, "use_mlx_vae", None),
        status=str(status)[-300:],
    )
    if not ok:
        return 2

    if os.environ.get("F0_SLIM") == "1":
        t = time.time()
        slim_after_init(dit)
        report("slim", t, use_mlx_dit=getattr(dit, "use_mlx_dit", None))

    replay = os.environ.get("F0_REPLAY")
    tag = os.environ.get("F0_TAG", "run")
    save_dir = os.path.join(OUT, tag)
    config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[SEED], audio_format="flac")
    if replay:
        llm = None
        params = fixed_params(replay, GenerationParams)
    else:
        t = time.time()
        llm = LLMHandler()
        lm_status, lm_ok = llm.initialize(CKPT, "acestep-5Hz-lm-0.6B", backend="mlx", device="auto")
        report("init_lm", t, ok=lm_ok, backend=str(getattr(llm, "llm_backend", "?")),
               status=str(lm_status)[-200:])
        if not lm_ok:
            return 3
        t = time.time()
        sample = create_sample(llm, QUERY)
        report("create_sample", t, ok=getattr(sample, "success", None),
               bpm=sample.bpm, keyscale=sample.keyscale, language=sample.language,
               lm_duration=sample.duration, caption=(sample.caption or "")[:160],
               lyrics_lines=len((sample.lyrics or "").splitlines()))
        params = GenerationParams(
            caption=sample.caption, lyrics=sample.lyrics, bpm=sample.bpm, duration=DURATION,
            keyscale=sample.keyscale, vocal_language=sample.language or "unknown", thinking=True,
        )

    t = time.time()
    result = generate_music(dit, llm, params, config, save_dir=save_dir)
    paths = [a.get("path") for a in result.audios] if result.success else []
    report("generate_music", t, ok=result.success, error=result.error, paths=paths,
           replay=bool(replay), torch_dtype=str(getattr(dit, "dtype", "?")),
           use_mlx_dit=getattr(dit, "use_mlx_dit", None), use_mlx_vae=getattr(dit, "use_mlx_vae", None))
    if result.success and os.environ.get("F0_SAVE_FIXED"):
        save_fixed(os.environ["F0_SAVE_FIXED"], result.audios[0])
    return 0 if result.success else 4


if __name__ == "__main__":
    sys.exit(main())
