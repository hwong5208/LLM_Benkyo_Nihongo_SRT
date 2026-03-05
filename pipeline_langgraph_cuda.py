"""
pipeline_langgraph_cuda.py — LangGraph entrypoint for NVIDIA CUDA (Ollama).

This script:
  1. Verifies Ollama is running and the model is available
  2. Injects Ollama-based inference functions into nodes.py
  3. Builds and compiles the LangGraph StateGraph
  4. Runs the full pipeline via app.invoke()

Usage:
    python pipeline_langgraph_cuda.py --input sample.mp4 --output-dir ./output
"""

import os
import sys
import re
import gc
import json
import time
import logging
import argparse
import subprocess
from datetime import timedelta
from typing import List, Optional

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Silence noisy internal debug loggers — only show our own DEBUG lines
for _noisy in ("httpcore", "httpx", "urllib3", "faster_whisper", "ctranslate2"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# --- CUDA / Faster-Whisper imports ---
# NOTE: WhisperModel is imported inside _whisper_worker() (subprocess) only.
try:
    import requests
except ImportError:
    logger.critical("Dependencies not found. Run: pip install requests")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- LangGraph import ---
try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import RetryPolicy
except ImportError:
    logger.critical("langgraph not found. Run: pip install langgraph")
    sys.exit(1)

from nodes import (
    PipelineState,
    set_whisper_backend,
    set_inference_backend,
    set_batch_size,
    extract_audio_node,
    transcribe_node,
    translate_node,
    write_srt_node,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WORKSPACE_DIR       = os.getenv("WORKSPACE_DIR",     "./workspace")
OLLAMA_API_URL      = os.getenv("OLLAMA_API_URL",    "http://localhost:11434/api/generate")
OLLAMA_MODEL        = os.getenv("OLLAMA_MODEL",      "qwen2.5:7b-instruct")
OLLAMA_BATCH_SIZE   = int(os.getenv("OLLAMA_BATCH_SIZE", "25"))
WHISPER_MODEL_SIZE  = os.getenv("WHISPER_MODEL_SIZE","medium")
WHISPER_DEVICE      = os.getenv("WHISPER_DEVICE",    "cuda")
WHISPER_COMPUTE_TYPE= os.getenv("WHISPER_COMPUTE_TYPE", "float16")

# Ensure Ollama is on PATH (Windows convenience)
_ollama_path = r"C:\Users\hwong\AppData\Local\Programs\Ollama"
if os.path.exists(_ollama_path):
    os.environ["PATH"] += f";{_ollama_path}"


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def check_dependencies() -> bool:
    import subprocess
    try:
        subprocess.run(["ffmpeg", "-version"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.error("FFmpeg not found in PATH.")
        return False

    try:
        tags_url = OLLAMA_API_URL.replace("/api/generate", "/api/tags")
        tags = requests.get(tags_url, timeout=5).json()
        models = [m.get("name", "") for m in tags.get("models", [])]
        if not any(OLLAMA_MODEL in m for m in models):
            logger.error(f"Model '{OLLAMA_MODEL}' not in Ollama. Run: ollama pull {OLLAMA_MODEL}")
            return False
        logger.info(f"Ollama connection verified. Model '{OLLAMA_MODEL}' available.")
    except Exception as e:
        logger.error(f"Cannot reach Ollama at {OLLAMA_API_URL}: {e}")
        return False

    return True


# ---------------------------------------------------------------------------
# CUDA Whisper backend (Faster-Whisper) — runs in a SUBPROCESS
# ---------------------------------------------------------------------------
# Running Whisper in a child process completely solves the CUDA destructor
# crash: when the subprocess exits, the OS reclaims ALL GPU memory instantly.
# No Python destructor, no ctranslate2 double-free, no segfault.
# ---------------------------------------------------------------------------

import multiprocessing


def _log_vram(label: str):
    """Log current CUDA VRAM via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        used, free, total = [int(x.strip()) for x in result.stdout.strip().split(",")]
        logger.debug(
            f"[VRAM] {label}: "
            f"used={used}MB | free={free}MB | total={total}MB"
        )
    except Exception as e:
        logger.debug(f"[VRAM] {label}: nvidia-smi unavailable ({e})")


def _whisper_worker(audio_path, model_size, device, compute_type, segments_cache_path):
    """
    Whisper transcription worker — runs in a child process.

    Loads the model, transcribes, saves segments to disk, then exits.
    When this process exits, the OS frees ALL CUDA memory — no destructor needed.
    """
    # Self-contained imports so the child process is independent
    import json
    import sys
    import logging

    from faster_whisper import WhisperModel

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("whisper_worker")

    log.info(f"Faster-Whisper: Loading {model_size} on {device} ({compute_type})...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments_iter, info = model.transcribe(
        audio_path,
        beam_size=5,
        language="ja",
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )
    log.info(f"Detected language: {info.language} ({info.language_probability:.2f})")

    segments = []
    seen_text = None
    print("Transcribing: ", end="", flush=True)

    for seg in segments_iter:
        text = seg.text.strip()
        if not text or text in ["-", ".", "…", "。"]:
            continue
        if text == seen_text:
            print("x", end="", flush=True)
            continue
        seen_text = text
        segments.append({
            "index":   len(segments),
            "start":   seg.start,
            "end":     seg.end,
            "content": text,
        })
        print(".", end="", flush=True)

    print("\n", flush=True)
    log.info(f"Faster-Whisper: {len(segments)} clean segments.")

    # Write cache — the main process reads this after we exit
    log.info(f"Cache: Saving {len(segments)} segments to {segments_cache_path}")
    with open(segments_cache_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False)

    # Process exits here → OS frees ALL CUDA memory. No destructor needed.


def _cuda_whisper(audio_path: str) -> List[dict]:
    """
    Launch Whisper in a subprocess, wait for it, read segments from cache.
    VRAM is freed automatically when the child process exits.
    """
    segments_cache = audio_path + ".segments.json"

    _log_vram("before Whisper subprocess")

    p = multiprocessing.Process(
        target=_whisper_worker,
        args=(audio_path, WHISPER_MODEL_SIZE, WHISPER_DEVICE,
              WHISPER_COMPUTE_TYPE, segments_cache),
    )
    p.start()
    p.join()

    _log_vram("after Whisper subprocess exited (VRAM should be freed)")

    # The subprocess may exit with a non-zero code due to ctranslate2's
    # native destructor segfaulting during Python exit cleanup (0xC0000409
    # on Windows). This is harmless — the work is done and the cache file
    # is already on disk. Only raise if the cache file is missing/empty.
    if not os.path.exists(segments_cache) or os.path.getsize(segments_cache) == 0:
        raise RuntimeError(
            f"Whisper subprocess failed (exit code {p.exitcode}). "
            f"No segments cache produced."
        )

    if p.exitcode != 0:
        logger.warning(
            f"Whisper subprocess exited with code {p.exitcode} "
            f"(likely ctranslate2 destructor crash — harmless, data saved)."
        )

    with open(segments_cache, "r", encoding="utf-8") as f:
        segments = json.load(f)

    logger.info(f"Whisper subprocess complete: {len(segments)} segments loaded from cache.")
    return segments



# ---------------------------------------------------------------------------
# Ollama LLM inference backends
# ---------------------------------------------------------------------------

def _ollama_infer_single(text: str) -> str:
    prompt = (
        f"Translate the following Japanese subtitle into Traditional Chinese "
        f"(Taiwan/Hong Kong style). Output ONLY the translated text. Do not explain.\n\n"
        f"Japanese: {text}\n"
        f"Chinese:"
    )
    try:
        resp = requests.post(OLLAMA_API_URL, json={
            "model": OLLAMA_MODEL, "prompt": prompt, "stream": False
        })
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        logger.error(f"Single inference error: {e}")
        return text


def _ollama_infer_batch(lines: List[str]) -> Optional[List[str]]:
    prompt_text = "\n".join([f"{i+1}. {line}" for i, line in enumerate(lines)])
    prompt = (
        f"Translate the following {len(lines)} Japanese lines into Traditional Chinese "
        f"(Taiwan/Hong Kong style). Output ONLY the translated lines as a numbered list "
        f"(1., 2., etc.). Do not explain.\n\n{prompt_text}\nOutput:"
    )
    t0 = time.time()
    try:
        resp = requests.post(OLLAMA_API_URL, json={
            "model": OLLAMA_MODEL, "prompt": prompt, "stream": False
        })
        resp.raise_for_status()
        elapsed = time.time() - t0
        result = resp.json().get("response", "").strip()

        translated = []
        current_idx = 1
        for raw in result.split("\n"):
            raw = raw.strip()
            if not raw:
                continue
            match = re.match(r"^(\d+)[\.:、]\s*(.*)", raw)
            if match and int(match.group(1)) == current_idx:
                translated.append(match.group(2).strip())
                current_idx += 1

        if len(translated) != len(lines):
            logger.debug(
                f"[MISMATCH] elapsed={elapsed:.1f}s | "
                f"expected={len(lines)} got={len(translated)} | "
                f"raw_response_preview={repr(result[:400])}"
            )
            return None

        logger.debug(f"[BATCH OK] {len(lines)} lines in {elapsed:.1f}s")
        return translated
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"Batch inference error after {elapsed:.1f}s: {e}")
        return None


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph():
    """Wire and compile the LangGraph StateGraph for the CUDA platform."""

    set_whisper_backend(_cuda_whisper)
    set_inference_backend(_ollama_infer_single, _ollama_infer_batch)
    set_batch_size(OLLAMA_BATCH_SIZE)

    workflow = StateGraph(PipelineState)

    workflow.add_node("extract_audio", extract_audio_node)
    workflow.add_node("transcribe",    transcribe_node)
    workflow.add_node(
        "translate",
        translate_node,
        retry=RetryPolicy(max_attempts=3, backoff_factor=1.0),
    )
    workflow.add_node("write_srt",     write_srt_node)

    workflow.add_edge(START,           "extract_audio")
    workflow.add_edge("extract_audio", "transcribe")
    workflow.add_edge("transcribe",    "translate")
    workflow.add_edge("translate",     "write_srt")
    workflow.add_edge("write_srt",     END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Nihongo-SRT LangGraph Pipeline (CUDA)")
    parser.add_argument("--input",       required=True, help="Path to the input video or audio file")
    parser.add_argument("--output-dir",  default=".",   help="Directory to write .srt files")
    parser.add_argument("--print-graph", action="store_true", help="Print graph topology and exit")
    args = parser.parse_args()

    app = build_graph()

    if args.print_graph:
        app.get_graph().print_ascii()
        return

    if not check_dependencies():
        sys.exit(1)

    input_file  = os.path.abspath(args.input)
    output_dir  = os.path.abspath(args.output_dir)
    base_name   = os.path.splitext(os.path.basename(input_file))[0]
    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    if not os.path.exists(input_file):
        logger.critical(f"Input file not found: {input_file}")
        sys.exit(1)

    logger.info("=" * 50)
    logger.info(" Nihongo-SRT — LangGraph Pipeline (CUDA)")
    logger.info("=" * 50)
    logger.info(f"Input      : {input_file}")
    logger.info(f"Output Dir : {output_dir}")
    logger.info(f"Ollama URL : {OLLAMA_API_URL}")
    logger.info(f"LLM Model  : {OLLAMA_MODEL}")

    logger.info("Graph topology:")
    app.get_graph().print_ascii()

    initial_state: PipelineState = {
        "video_path":    input_file,
        "audio_path":    "",
        "segments":      [],
        "translated_map": {},
        "output_dir":    output_dir,
        "base_name":     base_name,
        "workspace_dir": WORKSPACE_DIR,
    }

    t0 = time.time()
    try:
        app.invoke(initial_state)
        elapsed = time.time() - t0
        logger.info(f"SUCCESS! Total time: {str(timedelta(seconds=int(elapsed)))}")
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
