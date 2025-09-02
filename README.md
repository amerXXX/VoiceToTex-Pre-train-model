# VoiceToTex-Pre-train-model
This project provides a Python script to transcribe audio and video files into plain text (.txt) and subtitle (.srt) formats using the faster-whisper library. Faster Whisper is a faster and more memory-efficient implementation of OpenAI's Whisper model.

# Features
Transcribes audio from various formats (e.g., .m4a, .mp3) supported by FFmpeg.

Generates output in both plain text (.txt) and SubRip Subtitle (.srt) formats.

Supports different Whisper model sizes for varying accuracy and speed trade-offs.

Allows selection of device (CPU or GPU) for transcription.

Includes an option for Voice Activity Detection (VAD) filtering to improve results.

# expected output format:
```
[i] Loading model='medium' on device='cuda' ...
[i] Transcribing: Recording 1.m4a
[✓] Done. Segments: 1294  ~2799s audio
[→] Transcript (plain text): /content/Recording 1.txt
[→] Subtitles (SRT):        /content/Recording 1.srt
