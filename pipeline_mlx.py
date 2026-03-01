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

# Set up logging for observability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Configuration for MLX (Apple Silicon) ---
try:
    import mlx_whisper
    from mlx_lm import load, generate
except ImportError:
    logger.critical("MLX modules not found. Please install: pip install mlx-whisper mlx-lm")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("python-dotenv not installed. Falling back to native environment variables.")

WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "/tmp/llm_srt_work")
WHISPER_MODEL = os.getenv("MLX_WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")
LLM_MODEL_PATH = os.getenv("MLX_LLM_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")
BATCH_SIZE = int(os.getenv("MLX_BATCH_SIZE", "20"))


# Global LLM variables
llm_model = None
llm_tokenizer = None

def ensure_workspace():
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    return WORKSPACE_DIR

def check_ffmpeg_path():
    try:
        if os.path.exists("/opt/homebrew/bin/ffmpeg"):
            return "/opt/homebrew/bin/ffmpeg"
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "ffmpeg"
    except FileNotFoundError:
        return None

def load_llm_if_needed():
    global llm_model, llm_tokenizer
    if llm_model is None:
        logger.info(f"MLX LLM: Loading {LLM_MODEL_PATH}...")
        llm_model, llm_tokenizer = load(LLM_MODEL_PATH)
        logger.info("MLX LLM: Model loaded.")

def extract_audio(video_path, ffmpeg_bin="ffmpeg", workspace=WORKSPACE_DIR):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(workspace, f"{base_name}.mp3")
    
    if os.path.exists(audio_path):
        logger.info(f"Skip Extract: Found existing audio: {audio_path}")
        return audio_path

    logger.info(f"FFmpeg: Extracting audio to {audio_path}...")
    cmd = [
        ffmpeg_bin, "-y", "-i", video_path,
        "-vn", "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1",
        audio_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path

def transcribe_audio(audio_path):
    logger.info(f"MLX Whisper: Transcribing with {WHISPER_MODEL}...")
    
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=WHISPER_MODEL,
        language="ja",
        hallucination_silence_threshold=2.0,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=False
    )
    
    segments = result.get('segments', [])
    srt_segments = []
    
    logger.info(f"Raw segments returned: {len(segments)}")
    
    print("Transcribing: ", end="", flush=True)
    for segment in segments:
        text = segment.get('text', '').strip()
        start_time = segment.get('start', 0.0)
        end_time = segment.get('end', 0.0)
        
        if not text or text in ["-", ".", "…"]:
            continue
            
        if srt_segments and srt_segments[-1].content == text:
            print("x", end="", flush=True)
            continue

        srt_segments.append(srt.Subtitle(
            index=0, 
            start=timedelta(seconds=start_time),
            end=timedelta(seconds=end_time),
            content=text
        ))
        print(".", end="", flush=True)
    print("\n", flush=True)
    
    logger.info("MLX Whisper: Transcription complete.")

    segments_cache = audio_path + ".segments.json"
    logger.info(f"Cache: Saving segments to {segments_cache}...")
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

    gc.collect()
    return srt_segments

def run_mlx_inference(prompt):
    load_llm_if_needed()
    
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = llm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    response = generate(
        llm_model, 
        llm_tokenizer, 
        prompt=formatted_prompt, 
        verbose=False, 
        max_tokens=2048
    )
    return response.strip()

def translate_text_mlx(text):
    prompt = (
        f"Translate the following Japanese subtitle into Traditional Chinese (Taiwan/Hong Kong style). "
        f"Output ONLY the translated text. Do not explain.\n\n"
        f"Japanese: {text}\n"
        f"Chinese:"
    )
    try:
        return run_mlx_inference(prompt)
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text 

def translate_batch_mlx(lines):
    prompt_text = "\n".join([f"{i+1}. {line}" for i, line in enumerate(lines)])
    prompt = (
        f"Translate the following {len(lines)} Japanese lines into Traditional Chinese (Taiwan/Hong Kong style). "
        f"Output ONLY the translated lines as a numbered list (1., 2., etc.). Do not explain.\n\n"
        f"{prompt_text}\n"
        f"Output:"
    )

    try:
        result = run_mlx_inference(prompt)
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
            return None 
        return translated_lines
    except Exception as e:
        logger.error(f"Batch Error: Request failed: {e}")
        return None

def generate_srts(original_segments, output_dir, base_filename):
    jp_srt_path = os.path.join(output_dir, base_filename + ".jp.srt")
    cn_srt_path = os.path.join(output_dir, base_filename + ".cn.srt")
    bi_srt_path = os.path.join(output_dir, base_filename + ".srt")

    trans_cache_path = os.path.join(WORKSPACE_DIR, base_filename + ".trans_cache.json")
    translated_map = {}
    
    if os.path.exists(trans_cache_path):
        logger.info(f"Resume: Found existing translation cache: {trans_cache_path}")
        try:
            with open(trans_cache_path, "r", encoding="utf-8") as f:
                translated_map = json.load(f)
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")

    jp_subs, cn_subs, bi_subs = [], [], []
    total_segments = len(original_segments)
    
    logger.info(f"Translate: Processing {total_segments} segments (Batch Size: {BATCH_SIZE})...")

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
            logger.info(f"  Batch Translating lines [{i+1}-{min(i+BATCH_SIZE, total_segments)}/{total_segments}]...")
            results = translate_batch_mlx(lines_to_translate)
            
            if results is None:
                logger.warning("  Fallback: Batch failed structure check, switching to single line...")
                results = [translate_text_mlx(line) for line in lines_to_translate]
            
            for local_idx, translated_text in enumerate(results):
                global_idx = indices_to_translate[local_idx]
                key = str(global_idx + 1)
                translated_map[key] = translated_text

            with open(trans_cache_path, "w", encoding="utf-8") as f:
                json.dump(translated_map, f, ensure_ascii=False, indent=2)
            gc.collect()

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

def format_duration(seconds):
    return str(timedelta(seconds=int(seconds)))

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Subtitle Generator (MLX)")
    parser.add_argument("--input", required=True, help="Path to the video or audio file")
    parser.add_argument("--output-dir", default=".", help="Directory to save generated SRTs")
    args = parser.parse_args()

    # Timing Stats
    stats = {"extract": 0, "whisper": 0, "llm": 0, "total": 0}
    script_start_time = time.time()

    input_file = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output_dir)
    
    if not os.path.exists(input_file):
        logger.critical(f"Error: Input path not found: {input_file}")
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Input File: {input_file}")
    logger.info(f"Target Output Directory : {output_dir}")
    
    workspace = ensure_workspace()

    ffmpeg_bin = check_ffmpeg_path()
    if not ffmpeg_bin:
        logger.critical("CRITICAL ERROR: 'ffmpeg' not found.")
        sys.exit(1)

    try:
        local_audio_path = os.path.join(workspace, base_name + ".mp3")
        segments_cache = local_audio_path + ".segments.json"
        
        # 1. OPTIMIZATION: Skip extraction if cached
        if os.path.exists(segments_cache):
            logger.info("Resume: Found segments cache, skipping audio extraction.")
        elif os.path.exists(local_audio_path):
             logger.info("Resume: Found existing audio.")
        else:
            t0 = time.time()
            local_audio_path = extract_audio(input_file, ffmpeg_bin, workspace) 
            stats["extract"] = time.time() - t0
            logger.info(f"Time - Extract: {format_duration(stats['extract'])}")
            
        # 2. Whisper
        t0 = time.time()
        if os.path.exists(segments_cache):
            logger.info(f"Skip Whisper: Found cached segments.")
            with open(segments_cache, "r", encoding="utf-8") as f:
                data = json.load(f)
                segments = [
                    srt.Subtitle(
                        index=item.get("index", 0),
                        start=timedelta(seconds=item["start"]),
                        end=timedelta(seconds=item["end"]),
                        content=item["content"]
                    ) for item in data
                ]
        else:
            segments = transcribe_audio(local_audio_path)
            
        stats["whisper"] = time.time() - t0
        logger.info(f"Time - Whisper: {format_duration(stats['whisper'])}")
        gc.collect()

        # 3. LLM Translate
        t0 = time.time()
        generated_files = generate_srts(segments, output_dir, base_name)
        stats["llm"] = time.time() - t0
        logger.info(f"Time - LLM: {format_duration(stats['llm'])}")
        
        logger.info("SUCCESS! All processing complete.")
        for f in generated_files:
            logger.info(f" - {os.path.basename(f)}")
            
    except Exception as e:
        logger.critical(f"CRITICAL ERROR: {e}", exc_info=True)
    
    finally:
        stats["total"] = time.time() - script_start_time
        logger.info("========================================")
        logger.info("          PERFORMANCE SUMMARY           ")
        logger.info("========================================")
        logger.info(f"Extract Audio  : {format_duration(stats['extract'])}")
        logger.info(f"Whisper (Trans): {format_duration(stats['whisper'])}")
        logger.info(f"LLM (Translate): {format_duration(stats['llm'])}")
        logger.info("----------------------------------------")
        logger.info(f"TOTAL TIME     : {format_duration(stats['total'])}")
        logger.info("========================================")

if __name__ == "__main__":
    main()
