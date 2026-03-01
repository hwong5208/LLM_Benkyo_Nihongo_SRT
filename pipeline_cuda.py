import os
import sys
import shutil
import subprocess
import argparse
import time
import json
import logging
import gc
import srt
from datetime import timedelta
from faster_whisper import WhisperModel
import requests

# Set up logging for observability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Configuration (Loaded via dotenv or environment) ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("python-dotenv not installed. Falling back to native environment variables.")

# If "ollama not found" on Windows, sometimes adding to PATH helps.
ollama_path = r"C:\Users\hwong\AppData\Local\Programs\Ollama"
if os.path.exists(ollama_path):
    os.environ["PATH"] += f";{ollama_path}"

WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "./workspace")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
BATCH_SIZE = int(os.getenv("OLLAMA_BATCH_SIZE", "25"))


def ensure_workspace():
    """Ensure the local scratch workspace exists."""
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    return WORKSPACE_DIR


def check_dependencies():
    """Verify system dependencies (FFmpeg, Ollama) are available."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        logger.error("FFmpeg not found in PATH. Please install FFmpeg.")
        return False
        
    logger.info(f"Verifying Ollama model: {OLLAMA_MODEL}...")
    try:
        tags = requests.get(OLLAMA_API_URL.replace("/api/generate", "/api/tags")).json()
        models = [m.get('name', '') for m in tags.get('models', [])]
        if not any(OLLAMA_MODEL in m for m in models):
            logger.error(f"Model '{OLLAMA_MODEL}' not found in Ollama locally. Please run: ollama pull {OLLAMA_MODEL}")
            return False
        logger.info("Ollama connection verified.")
    except Exception as e:
        logger.error(f"Could not connect to Ollama: {e}. Is 'ollama serve' running?")
        return False
        
    return True


def extract_audio(video_path, workspace):
    """Extract audio from the video file to the workspace."""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(workspace, f"{base_name}.mp3")
    
    if os.path.exists(audio_path):
        logger.info(f"Skip Extract: Found existing audio at {audio_path}")
        return audio_path

    logger.info(f"FFmpeg: Extracting audio to {audio_path}...")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1",
        audio_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path


def transcribe_audio(audio_path):
    """Transcribe the audio using Faster-Whisper, with checkpointing."""
    segments_cache = audio_path + ".segments.json"
    
    if os.path.exists(segments_cache):
        logger.info(f"Resume: Found cached Whisper segments at {segments_cache}")
        with open(segments_cache, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [
                srt.Subtitle(
                    index=item["index"],
                    start=timedelta(seconds=item["start"]),
                    end=timedelta(seconds=item["end"]),
                    content=item["content"]
                ) for item in data
            ]

    logger.info(f"Whisper: Transcribing with {WHISPER_MODEL_SIZE} on {WHISPER_DEVICE} ({WHISPER_COMPUTE_TYPE})...")
    model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    
    # Using more aggressive hallucination filters similar to the MLX pipeline
    segments, info = model.transcribe(
        audio_path, 
        beam_size=5, 
        language="ja",
        condition_on_previous_text=False, # Reduces repetitive loops
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4
    )
    
    logger.info(f"Detected language: {info.language} (Probability: {info.language_probability:.2f})")
    
    srt_segments = []
    print("Transcribing: ", end="", flush=True)
    for segment in segments:
        text = segment.text.strip()
        
        # Drop empty strings or noise tokens
        if not text or text in ["-", ".", "…", "。"]:
            continue
            
        # Deduplicate identical consecutive segments (common in long silences)
        if srt_segments and srt_segments[-1].content == text:
            print("x", end="", flush=True) # Indicate skipped duplicate
            continue
            
        srt_segments.append(srt.Subtitle(
            index=0, 
            start=timedelta(seconds=segment.start),
            end=timedelta(seconds=segment.end),
            content=text
        ))
        print(".", end="", flush=True)
    print("\n", flush=True)

    # Save Cache Checkpoint
    logger.info(f"Cache: Checkpointing segments to {segments_cache}")
    data = [
        {
            "index": s.index,
            "start": s.start.total_seconds(),
            "end": s.end.total_seconds(),
            "content": s.content
        } for s in srt_segments
    ]
    with open(segments_cache, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    # Free VRAM
    del model
    gc.collect()
    time.sleep(2)

    return srt_segments


def translate_text(text):
    """Fallback single-line translation."""
    prompt = (
        f"Translate the following Japanese subtitle into Traditional Chinese (Taiwan/Hong Kong style). "
        f"Output ONLY the translated text. Do not explain.\n\n"
        f"Japanese: {text}\n"
        f"Chinese:"
    )
    try:
        response = requests.post(OLLAMA_API_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        })
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text 

def translate_batch(lines):
    """Batch translation via Ollama."""
    prompt_text = "\n".join([f"{i+1}. {line}" for i, line in enumerate(lines)])
    prompt = (
        f"Translate the following {len(lines)} Japanese lines into Traditional Chinese (Taiwan/Hong Kong style). "
        f"Output ONLY the translated lines as a numbered list (1., 2., etc.). Do not explain.\n\n"
        f"{prompt_text}\n"
        f"Output:"
    )

    try:
        response = requests.post(OLLAMA_API_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        })
        response.raise_for_status()
        result = response.json().get("response", "").strip()
        
        translated_lines = []
        raw_lines = result.split('\n')
        
        current_idx = 1
        import re
        for raw in raw_lines:
            raw = raw.strip()
            if not raw: continue
            match = re.match(r'^(\d+)[\.:、]\s*(.*)', raw)
            if match:
                idx = int(match.group(1))
                content = match.group(2).strip()
                if idx == current_idx:
                    translated_lines.append(content)
                    current_idx += 1
        
        if len(translated_lines) != len(lines):
            return None # Structural mismatch fail
        return translated_lines

    except Exception as e:
        logger.error(f"Batch Request failed: {e}")
        return None

def generate_srts(original_segments, output_dir, base_filename):
    """Translate segments and construct SRT files."""
    jp_srt_path = os.path.join(output_dir, base_filename + ".jp.srt")
    cn_srt_path = os.path.join(output_dir, base_filename + ".cn.srt")
    bi_srt_path = os.path.join(output_dir, base_filename + ".srt")
    
    trans_cache_path = os.path.join(WORKSPACE_DIR, base_filename + ".trans_cache.json")
    translated_map = {}
    
    if os.path.exists(trans_cache_path):
        logger.info(f"Resume: Loaded translation cache from {trans_cache_path}")
        with open(trans_cache_path, "r", encoding="utf-8") as f:
            translated_map = json.load(f)
            
    jp_subs, cn_subs, bi_subs = [], [], []
    total_segments = len(original_segments)
    
    logger.info(f"LLM: Translating {total_segments} segments (Batch Size: {BATCH_SIZE})")

    for i in range(0, total_segments, BATCH_SIZE):
        batch_segment_indices = range(i, min(i + BATCH_SIZE, total_segments))
        
        lines_to_translate = []
        indices_to_translate = []
        
        for idx in batch_segment_indices:
            key = str(idx + 1) 
            if key not in translated_map:
                lines_to_translate.append(original_segments[idx].content)
                indices_to_translate.append(idx)
        
        if lines_to_translate:
            logger.info(f"  Translating lines [{i+1}-{min(i+BATCH_SIZE, total_segments)}/{total_segments}]...")
            results = translate_batch(lines_to_translate)
            
            if results is None:
                logger.warning("  Batch structure mismatch. Falling back to sequential translation.")
                results = [translate_text(line) for line in lines_to_translate]
            
            for local_idx, translated_text in enumerate(results):
                global_idx = indices_to_translate[local_idx]
                key = str(global_idx + 1)
                translated_map[key] = translated_text

            # Checkpoint cache
            with open(trans_cache_path, "w", encoding="utf-8") as f:
                json.dump(translated_map, f, ensure_ascii=False, indent=2)

    # Reconstruct subtitles
    logger.info("Constructing dual-language subtitles...")
    for idx in range(total_segments):
        seg = original_segments[idx]
        index = idx + 1
        key = str(index)
        
        jp_subs.append(srt.Subtitle(index, seg.start, seg.end, seg.content))
        
        cn_text = translated_map.get(key, "")
        cn_subs.append(srt.Subtitle(index, seg.start, seg.end, cn_text))
        
        bi_content = f"{seg.content}\n{cn_text}"
        bi_subs.append(srt.Subtitle(index, seg.start, seg.end, bi_content))

    with open(jp_srt_path, "w", encoding="utf-8") as f: f.write(srt.compose(jp_subs))
    with open(cn_srt_path, "w", encoding="utf-8") as f: f.write(srt.compose(cn_subs))
    with open(bi_srt_path, "w", encoding="utf-8") as f: f.write(srt.compose(bi_subs))
        
    return [jp_srt_path, cn_srt_path, bi_srt_path]


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Subtitle Pipeline (CUDA)")
    parser.add_argument("--input", required=True, help="Path to input video or audio file")
    parser.add_argument("--output-dir", default=".", help="Directory to save generated SRTs")
    args = parser.parse_args()

    input_file = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output_dir)
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    if not os.path.exists(input_file):
        logger.critical(f"Input file not found: {input_file}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    workspace = ensure_workspace()

    logger.info(f"Starting Multi-Agent Pipeline for: {base_name}")
    logger.info(f"Target Output Directory: {output_dir}")

    if not check_dependencies():
        sys.exit(1)

    try:
        audio_path = extract_audio(input_file, workspace)
        segments = transcribe_audio(audio_path)
        generated_files = generate_srts(segments, output_dir, base_name)
        
        logger.info(f"SUCCESS! Pipeline complete. Output files:")
        for f in generated_files:
            logger.info(f" - {os.path.basename(f)}")
            
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
