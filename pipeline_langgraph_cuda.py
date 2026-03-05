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
import time
import logging
import argparse
from datetime import timedelta
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# --- CUDA / Faster-Whisper imports ---
try:
    from faster_whisper import WhisperModel
    import requests
except ImportError:
    logger.critical("Dependencies not found. Run: pip install faster-whisper requests")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- LangGraph import ---
try:
    from langgraph.graph import StateGraph, START, END
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
# CUDA Whisper backend (Faster-Whisper)
# ---------------------------------------------------------------------------

def _cuda_whisper(audio_path: str) -> List[dict]:
    """Run Faster-Whisper on CUDA, return list of segment dicts."""
    logger.info(f"Faster-Whisper: Loading {WHISPER_MODEL_SIZE} on {WHISPER_DEVICE} ({WHISPER_COMPUTE_TYPE})...")
    model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE,
                         compute_type=WHISPER_COMPUTE_TYPE)

    segments_iter, info = model.transcribe(
        audio_path,
        beam_size=5,
        language="ja",
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )
    logger.info(f"Detected language: {info.language} ({info.language_probability:.2f})")

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
    logger.info(f"Faster-Whisper: {len(segments)} clean segments.")

    # Free VRAM
    del model
    gc.collect()
    time.sleep(2)

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
    try:
        resp = requests.post(OLLAMA_API_URL, json={
            "model": OLLAMA_MODEL, "prompt": prompt, "stream": False
        })
        resp.raise_for_status()
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

        return translated if len(translated) == len(lines) else None
    except Exception as e:
        logger.error(f"Batch inference error: {e}")
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
    workflow.add_node("translate",     translate_node)
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
