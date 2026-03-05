"""
pipeline_langgraph_mlx.py — LangGraph entrypoint for Apple Silicon (MLX).

This script:
  1. Loads the MLX Whisper + Qwen2.5 models
  2. Injects the MLX inference functions into nodes.py
  3. Builds and compiles the LangGraph StateGraph
  4. Runs the full pipeline via app.invoke()

Usage:
    python pipeline_langgraph_mlx.py --input sample.mp4 --output-dir ./output
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

# --- MLX imports ---
try:
    import mlx_whisper
    from mlx_lm import load, generate
except ImportError:
    logger.critical("MLX modules not found. Run: pip install mlx-whisper mlx-lm")
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
WORKSPACE_DIR    = os.getenv("WORKSPACE_DIR", "/tmp/llm_srt_work")
WHISPER_MODEL    = os.getenv("MLX_WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")
LLM_MODEL_PATH   = os.getenv("MLX_LLM_MODEL",    "mlx-community/Qwen2.5-7B-Instruct-4bit")
MLX_BATCH_SIZE   = int(os.getenv("MLX_BATCH_SIZE", "20"))

# Lazy-loaded globals (avoids loading models until actually needed)
_llm_model     = None
_llm_tokenizer = None


def _load_llm():
    global _llm_model, _llm_tokenizer
    if _llm_model is None:
        logger.info(f"MLX LLM: Loading {LLM_MODEL_PATH}...")
        _llm_model, _llm_tokenizer = load(LLM_MODEL_PATH)
        logger.info("MLX LLM: Ready.")


# ---------------------------------------------------------------------------
# MLX Whisper backend
# ---------------------------------------------------------------------------

def _mlx_whisper(audio_path: str) -> List[dict]:
    """Run MLX Whisper transcription, return list of segment dicts."""
    logger.info(f"MLX Whisper: Transcribing with {WHISPER_MODEL}...")

    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=WHISPER_MODEL,
        language="ja",
        hallucination_silence_threshold=2.0,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=False,
    )

    raw_segments = result.get("segments", [])
    logger.info(f"MLX Whisper: {len(raw_segments)} raw segments returned.")

    segments = []
    seen_text = None
    print("Transcribing: ", end="", flush=True)

    for seg in raw_segments:
        text = seg.get("text", "").strip()
        if not text or text in ["-", ".", "…"]:
            continue
        if text == seen_text:
            print("x", end="", flush=True)
            continue
        seen_text = text
        segments.append({
            "index": len(segments),
            "start": seg.get("start", 0.0),
            "end":   seg.get("end",   0.0),
            "content": text,
        })
        print(".", end="", flush=True)

    print("\n", flush=True)
    logger.info(f"MLX Whisper: {len(segments)} clean segments.")

    # Write cache atomically before returning — crash-safe
    segments_cache = audio_path + ".segments.json"
    logger.info(f"Cache: Saving segments to {segments_cache}")
    with open(segments_cache, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False)

    gc.collect()
    return segments


# ---------------------------------------------------------------------------
# MLX LLM inference backends
# ---------------------------------------------------------------------------

def _mlx_infer_single(text: str) -> str:
    _load_llm()
    prompt = (
        f"Translate the following Japanese subtitle into Traditional Chinese "
        f"(Taiwan/Hong Kong style). Output ONLY the translated text. Do not explain.\n\n"
        f"Japanese: {text}\n"
        f"Chinese:"
    )
    messages = [{"role": "user", "content": prompt}]
    formatted = _llm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    response = generate(_llm_model, _llm_tokenizer, prompt=formatted,
                        verbose=False, max_tokens=2048)
    return response.strip()


def _mlx_infer_batch(lines: List[str]) -> Optional[List[str]]:
    _load_llm()
    import re as _re
    prompt_text = "\n".join([f"{i+1}. {line}" for i, line in enumerate(lines)])
    prompt = (
        f"Translate the following {len(lines)} Japanese lines into Traditional Chinese "
        f"(Taiwan/Hong Kong style). Output ONLY the translated lines as a numbered list "
        f"(1., 2., etc.). Do not explain.\n\n{prompt_text}\nOutput:"
    )
    messages = [{"role": "user", "content": prompt}]
    formatted = _llm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    try:
        result = generate(_llm_model, _llm_tokenizer, prompt=formatted,
                          verbose=False, max_tokens=2048)
        translated = []
        current_idx = 1
        for raw in result.split("\n"):
            raw = raw.strip()
            if not raw:
                continue
            match = _re.match(r"^(\d+)[\.:、]\s*(.*)", raw)
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
    """Wire and compile the LangGraph StateGraph for the MLX platform."""

    # Register platform-specific backends into nodes.py
    set_whisper_backend(_mlx_whisper)
    set_inference_backend(_mlx_infer_single, _mlx_infer_batch)
    set_batch_size(MLX_BATCH_SIZE)

    workflow = StateGraph(PipelineState)

    # Register nodes
    workflow.add_node("extract_audio", extract_audio_node)
    workflow.add_node("transcribe",    transcribe_node)
    workflow.add_node(
        "translate",
        translate_node,
        retry=RetryPolicy(max_attempts=3, backoff_factor=1.0),
    )
    workflow.add_node("write_srt",     write_srt_node)

    # Define edges (linear DAG)
    workflow.add_edge(START,          "extract_audio")
    workflow.add_edge("extract_audio","transcribe")
    workflow.add_edge("transcribe",   "translate")
    workflow.add_edge("translate",    "write_srt")
    workflow.add_edge("write_srt",    END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Nihongo-SRT LangGraph Pipeline (MLX)")
    parser.add_argument("--input",      required=True, help="Path to the input video or audio file")
    parser.add_argument("--output-dir", default=".",   help="Directory to write .srt files")
    parser.add_argument("--print-graph", action="store_true", help="Print graph topology and exit")
    args = parser.parse_args()

    app = build_graph()

    if args.print_graph:
        app.get_graph().print_ascii()
        return

    input_file  = os.path.abspath(args.input)
    output_dir  = os.path.abspath(args.output_dir)
    base_name   = os.path.splitext(os.path.basename(input_file))[0]
    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    if not os.path.exists(input_file):
        logger.critical(f"Input file not found: {input_file}")
        sys.exit(1)

    logger.info("=" * 50)
    logger.info(" Nihongo-SRT — LangGraph Pipeline (MLX)")
    logger.info("=" * 50)
    logger.info(f"Input      : {input_file}")
    logger.info(f"Output Dir : {output_dir}")

    # Print graph for observability
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
