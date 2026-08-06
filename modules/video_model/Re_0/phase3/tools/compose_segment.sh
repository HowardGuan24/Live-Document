#!/usr/bin/env bash
set -euo pipefail

SEGMENT_DIR="${1:?segment directory is required}"
INPUT="$SEGMENT_DIR/video.mp4"
START_OVERLAY="$SEGMENT_DIR/overlay_start.png"
END_OVERLAY="$SEGMENT_DIR/overlay_end.png"
SUBTITLES="$SEGMENT_DIR/subtitles.srt"
OUTPUT="$SEGMENT_DIR/annotated.mp4"

for path in "$INPUT" "$START_OVERLAY" "$END_OVERLAY" "$SUBTITLES"; do
  test -s "$path"
done

DURATION="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$INPUT")"
FADE_START="$(awk -v duration="$DURATION" 'BEGIN { printf "%.3f", duration * 0.42 }')"

ffmpeg -y -loglevel error \
  -i "$INPUT" \
  -loop 1 -i "$START_OVERLAY" \
  -loop 1 -i "$END_OVERLAY" \
  -filter_complex "[1:v]scale=512:288,format=rgba,fade=t=out:st=${FADE_START}:d=0.35:alpha=1[os];[2:v]scale=512:288,format=rgba,fade=t=in:st=${FADE_START}:d=0.35:alpha=1[oe];[0:v][os]overlay=shortest=1[tmp];[tmp][oe]overlay=shortest=1,subtitles=${SUBTITLES}:force_style='FontName=Noto Sans CJK SC,FontSize=14,PrimaryColour=&H00FFFFFF,OutlineColour=&H90000000,BorderStyle=1,Outline=1,Shadow=0,MarginV=12'[v]" \
  -map "[v]" -map 0:a? \
  -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
  -c:a aac -b:a 128k -shortest -movflags +faststart \
  "$OUTPUT"
