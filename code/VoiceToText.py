# 1) Python packages
!pip install faster-whisper tqdm

# 2) FFmpeg (required so Python can read .m4a, .mp3, etc.)
# macOS (Homebrew):
!brew install ffmpeg
# Ubuntu/Debian:
!sudo apt-get update && sudo apt-get install -y ffmpeg
# Windows:
# - Install FFmpeg from https://www.gyan.dev/ffmpeg/builds/ and add ffmpeg/bin to PATH

#runing
import argparse
from pathlib import Path
from datetime import timedelta
from tqdm import tqdm
from faster_whisper import WhisperModel
import sys
import torch

def format_timestamp(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    # SRT needs , for milliseconds
    return str(td)[:-3].replace(".", ",")

def write_srt(segments, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n")
            f.write(seg.text.strip() + "\n\n")

def write_txt(segments, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(seg.text.strip() + " ")

def main(audio_file_path: str):
    # Remove the -f argument added by Colab's kernel runner if it exists
    if "-f" in sys.argv:
        sys.argv.remove("-f")

    ap = argparse.ArgumentParser(description="Transcribe audio to TXT and SRT using faster-whisper.")
    ap.add_argument("audio", help="Path to audio/video file (e.g., .m4a from Notability).")
    ap.add_argument("--model", default="medium", help="Model size: tiny|base|small|medium|large-v3 (default: medium)")
    ap.add_argument("--device", default="auto", help="Device: auto|cpu|cuda|metal (Apple Silicon) (default: auto)")
    ap.add_argument("--compute-type", default='float16',
                    help="Precision: e.g., float16 (GPU), int8_float16 (CPU), int8 (CPU). If omitted, auto-chooses.")
    ap.add_argument("--language", default="en", help="Language code (default: en).")
    ap.add_argument("--beam-size", type=int, default=5, help="Beam search size (default: 5)")
    ap.add_argument("--vad-filter", action="store_true", help="Enable voice-activity detection to reduce noise.")
    args = ap.parse_args([audio_file_path]) # Pass the audio file path explicitly

    audio_path = Path(args.audio)
    assert audio_path.exists(), f"File not found: {audio_path}"

    # Pick some sensible defaults for performance vs quality
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = WhisperModel(
        args.model,
        device=device,
        compute_type=args.compute_type,  # let faster-whisper auto-pick if None
    )

    print(f"[i] Loading model='{args.model}' on device='{device}' ...")
    print(f"[i] Transcribing: {audio_path.name}")

    segments, info = model.transcribe(
        str(audio_path),
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=args.vad_filter,
        condition_on_previous_text=True
    )

    # We need to iterate once to materialize segments for writing / progress
    segs = list(segments)

    # Outputs
    out_base = audio_path.with_suffix("")  # remove .m4a -> base path
    txt_path = out_base.with_suffix(".txt")
    srt_path = out_base.with_suffix(".srt")

    # Write files
    write_txt(segs, txt_path)
    write_srt(segs, srt_path)

    # Simple progress/info
    total_dur = sum(max(0.0, s.end - s.start) for s in segs)
    print(f"[✓] Done. Segments: {len(segs)}  ~{int(total_dur)}s audio")
    print(f"[→] Transcript (plain text): {txt_path}")
    print(f"[→] Subtitles (SRT):        {srt_path}")

if __name__ == "__main__":
    # Replace 'your_audio_file.m4a' with the actual path to your audio file
    audio_file = "/content/Recording 1.m4a"
    main(audio_file)
