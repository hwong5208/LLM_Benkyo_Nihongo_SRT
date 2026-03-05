"""
nodes.py — Shared LangGraph node definitions for the Nihongo-SRT pipeline.

This module is platform-agnostic. It defines:
  - PipelineState: the single state object passed through the graph.
  - Node functions: each takes state, returns a partial state update dict.

Platform-specific inference (MLX vs Ollama) is injected at runtime via the
`set_inference_backend(...)` helper before the graph is compiled.
"""

import os
import re
import gc
import json
import logging
import subprocess
import sys

import srt
from datetime import timedelta
from typing import TypedDict, List, Optional, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    """The single shared state object that flows through every graph node."""
    video_path: str          # Absolute path to the input video/audio file
    audio_path: str          # Path to the extracted .mp3 (set by extract_audio_node)
    segments: List[dict]     # Whisper output: list of {index, start, end, content}
    translated_map: dict     # LLM output: {"1": "中文...", "2": "..."} keyed by str(idx+1)
    output_dir: str          # Where to write the final .srt files
    base_name: str           # Filename stem (e.g. "episode01")
    workspace_dir: str       # Scratch workspace for caches


# ---------------------------------------------------------------------------
# Inference backend injection
# ---------------------------------------------------------------------------
# Each platform entrypoint calls set_inference_backend() to provide the two
# inference callables before compile()-ing the graph.

_infer_single_fn: Optional[Callable[[str], str]] = None   # translate one line
_infer_batch_fn: Optional[Callable[[List[str]], Optional[List[str]]]] = None  # translate batch


def set_inference_backend(
    single_fn: Callable[[str], str],
    batch_fn: Callable[[List[str]], Optional[List[str]]],
):
    """
    Inject platform-specific inference callables.

    Args:
        single_fn:  fn(japanese_text: str) -> chinese_text: str
        batch_fn:   fn(lines: List[str]) -> List[str] | None
                    Returns None on structural mismatch (triggers fallback).
    """
    global _infer_single_fn, _infer_batch_fn
    _infer_single_fn = single_fn
    _infer_batch_fn = batch_fn
    logger.info("Inference backend registered.")


# ---------------------------------------------------------------------------
# Node 1 — Extract Audio
# ---------------------------------------------------------------------------

def extract_audio_node(state: PipelineState) -> dict:
    """
    Node: Extract audio from the input video using FFmpeg.
    Skips extraction if the .mp3 already exists in workspace (resume support).
    Updates: audio_path
    """
    video_path = state["video_path"]
    workspace_dir = state["workspace_dir"]
    base_name = state["base_name"]
    audio_path = os.path.join(workspace_dir, f"{base_name}.mp3")

    if os.path.exists(audio_path):
        logger.info(f"Skip Extract: Found existing audio: {audio_path}")
        return {"audio_path": audio_path}

    logger.info(f"FFmpeg: Extracting audio to {audio_path}...")

    # Resolve ffmpeg binary (macOS Homebrew path or system PATH)
    ffmpeg_bin = "ffmpeg"
    if os.path.exists("/opt/homebrew/bin/ffmpeg"):
        ffmpeg_bin = "/opt/homebrew/bin/ffmpeg"

    cmd = [
        ffmpeg_bin, "-y", "-i", video_path,
        "-vn", "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1",
        audio_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.info(f"FFmpeg: Extraction complete → {audio_path}")
    return {"audio_path": audio_path}


# ---------------------------------------------------------------------------
# Node 2 — Transcribe Audio (Whisper)
# ---------------------------------------------------------------------------

def transcribe_node(state: PipelineState) -> dict:
    """
    Node: Transcribe audio with Whisper.
    Resume: If a segments cache JSON already exists, load it and skip inference.
    Updates: segments (list of serializable dicts)

    NOTE: The actual whisper inference callable is injected by the platform
    entrypoint and called via _run_whisper(audio_path). The platform module
    must call set_whisper_backend() before graph compilation.
    """
    audio_path = state["audio_path"]
    segments_cache = audio_path + ".segments.json"

    if os.path.exists(segments_cache):
        logger.info(f"Skip Whisper: Loading cached segments from {segments_cache}")
        with open(segments_cache, "r", encoding="utf-8") as f:
            segments = json.load(f)
        return {"segments": segments}

    # Delegate to the platform-specific whisper function
    segments = _run_whisper(audio_path)

    logger.info(f"Cache: Saving {len(segments)} segments to {segments_cache}")
    with open(segments_cache, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False)

    gc.collect()
    return {"segments": segments}


_whisper_fn: Optional[Callable[[str], List[dict]]] = None


def set_whisper_backend(fn: Callable[[str], List[dict]]):
    """
    Inject platform-specific whisper callable.
    fn(audio_path: str) -> List[dict]
      where each dict is: {index, start (float seconds), end (float), content (str)}
    """
    global _whisper_fn
    _whisper_fn = fn
    logger.info("Whisper backend registered.")


def _run_whisper(audio_path: str) -> List[dict]:
    if _whisper_fn is None:
        raise RuntimeError("No Whisper backend registered. Call set_whisper_backend() first.")
    return _whisper_fn(audio_path)


# ---------------------------------------------------------------------------
# Node 3 — Translate Segments (LLM)
# ---------------------------------------------------------------------------

_batch_size: int = 20  # Default; overridden by set_batch_size() at runtime


def set_batch_size(n: int):
    """Inject the platform-specific batch size before graph compilation."""
    global _batch_size
    _batch_size = n
    logger.info(f"Batch size set to {n}.")



def translate_node(state: PipelineState) -> dict:
    """
    Node: Translate all Japanese segments to Traditional Chinese using the LLM.
    Resume: Loads any existing translation cache and only translates missing entries.
    Updates: translated_map
    """
    if _infer_single_fn is None or _infer_batch_fn is None:
        raise RuntimeError("No inference backend registered. Call set_inference_backend() first.")

    segments = state["segments"]
    workspace_dir = state["workspace_dir"]
    base_name = state["base_name"]

    trans_cache_path = os.path.join(workspace_dir, base_name + ".trans_cache.json")
    translated_map: dict = {}

    if os.path.exists(trans_cache_path):
        logger.info(f"Resume: Loading translation cache from {trans_cache_path}")
        try:
            with open(trans_cache_path, "r", encoding="utf-8") as f:
                translated_map = json.load(f)
        except Exception as e:
            logger.warning(f"Cache load failed, starting fresh: {e}")

    total = len(segments)
    logger.info(f"Translate: {total} segments (batch size={_batch_size})")

    for i in range(0, total, _batch_size):
        batch_indices = list(range(i, min(i + _batch_size, total)))

        lines_to_translate = []
        indices_to_translate = []

        for idx in batch_indices:
            key = str(idx + 1)
            if key not in translated_map:
                lines_to_translate.append(segments[idx]["content"])
                indices_to_translate.append(idx)

        if not lines_to_translate:
            continue  # All already cached

        logger.info(f"  Translating [{i+1}–{min(i+_batch_size, total)}/{total}]...")
        results = _infer_batch_fn(lines_to_translate)

        if results is None:
            logger.warning("  Batch mismatch — falling back to single-line mode")
            results = [_infer_single_fn(line) for line in lines_to_translate]

        for local_idx, translated_text in enumerate(results):
            global_idx = indices_to_translate[local_idx]
            translated_map[str(global_idx + 1)] = translated_text

        # Checkpoint after every batch
        with open(trans_cache_path, "w", encoding="utf-8") as f:
            json.dump(translated_map, f, ensure_ascii=False, indent=2)

        gc.collect()

    return {"translated_map": translated_map}


# ---------------------------------------------------------------------------
# Node 4 — Write SRT Files
# ---------------------------------------------------------------------------

def write_srt_node(state: PipelineState) -> dict:
    """
    Node: Synthesize and write the final .jp.srt, .cn.srt, and bilingual .srt files.
    Returns empty dict — this is the terminal node with no state updates needed.
    """
    segments = state["segments"]
    translated_map = state["translated_map"]
    output_dir = state["output_dir"]
    base_name = state["base_name"]

    jp_srt_path = os.path.join(output_dir, base_name + ".jp.srt")
    cn_srt_path = os.path.join(output_dir, base_name + ".cn.srt")
    bi_srt_path = os.path.join(output_dir, base_name + ".srt")

    jp_subs, cn_subs, bi_subs = [], [], []

    for idx, seg in enumerate(segments):
        index = idx + 1
        key = str(index)
        start = timedelta(seconds=seg["start"])
        end = timedelta(seconds=seg["end"])
        jp_text = seg["content"]
        cn_text = translated_map.get(key, "")

        jp_subs.append(srt.Subtitle(index, start, end, jp_text))
        cn_subs.append(srt.Subtitle(index, start, end, cn_text))
        bi_subs.append(srt.Subtitle(index, start, end, f"{jp_text}\n{cn_text}"))

    os.makedirs(output_dir, exist_ok=True)

    with open(jp_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(jp_subs))
    with open(cn_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(cn_subs))
    with open(bi_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(bi_subs))

    logger.info(f"Output: {os.path.basename(jp_srt_path)}")
    logger.info(f"Output: {os.path.basename(cn_srt_path)}")
    logger.info(f"Output: {os.path.basename(bi_srt_path)}")
    return {}
