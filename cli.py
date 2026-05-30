"""
cli.py
=======
Command-line interface for the Complaint Auto-Routing System.

Usage:
  python cli.py --text "Water pipe burst near the market"
  python cli.py --audio path/to/complaint.wav
  python cli.py --video path/to/complaint.mp4
"""

import argparse
import sys
import os
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


PRIORITY_COLORS = {
    "High":   "\033[91m",   # red
    "Medium": "\033[93m",   # yellow
    "Low":    "\033[92m",   # green
}
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
BLUE   = "\033[94m"
YELLOW = "\033[93m"


def color(text, code):
    return f"{code}{text}{RESET}"


def print_header():
    print()
    print(color("=" * 62, CYAN))
    print(color("  🏛  Complaint Auto-Routing System  |  Municipal AI", BOLD + CYAN))
    print(color("=" * 62, CYAN))


def print_result(result: dict, input_text: str, lang_code: str = None):
    officer  = result["officer"]
    conf     = result["officer_confidence"]
    priority = result["priority"]
    pri_conf = result["priority_confidence"]
    eta      = result["eta_days"]
    similar  = result["similar_complaints"]
    xai      = result["explanation"]

    print()
    print(color("── Input ─────────────────────────────────────────────", BLUE))
    wrapped = textwrap.fill(input_text, width=58)
    for line in wrapped.splitlines():
        print(f"  {line}")
    if lang_code:
        print(f"  (Detected language: {lang_code})")

    print()
    print(color("── Prediction Results ────────────────────────────────", BLUE))

    pri_color = PRIORITY_COLORS.get(priority, "")
    print(f"  🏛  Officer  : {color(BOLD + officer, BOLD)}  ({conf:.1f}% confidence)")
    print(f"  🚨 Priority  : {color(priority, pri_color + BOLD)}  ({pri_conf:.1f}% confidence)")
    print(f"  ⏱  ETA      : {eta} day{'s' if eta != 1 else ''} estimated resolution")

    # Confidence breakdown
    print()
    print(color("── Officer Confidence Breakdown ──────────────────────", BLUE))
    all_probs = result.get("all_officer_probs", {})
    for off, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        bar_len = int(prob / 5)
        bar     = "█" * bar_len + "░" * (20 - bar_len)
        marker  = " ◄" if off == officer else ""
        print(f"  {off:<25} {bar} {prob:5.1f}%{marker}")

    # Top similar complaints
    print()
    print(color("── Top 5 Similar Past Complaints ─────────────────────", BLUE))
    for i, s in enumerate(similar[:5], 1):
        sim_text = s["text"][:72] + "…" if len(s["text"]) > 72 else s["text"]
        p_color  = PRIORITY_COLORS.get(s["priority"], "")
        print(f"  {i}. [{color(s['similarity'], YELLOW)}%] {sim_text}")
        print(f"     → {s['officer']} | "
              f"{color(s['priority'], p_color)} | {s['eta_days']} days")

    # XAI explanation
    print()
    print(color("── AI Explanation ────────────────────────────────────", BLUE))
    for line in xai.splitlines():
        clean_line = line.replace("**", "").replace("*", "")
        print(f"  {clean_line}")

    print()
    print(color("=" * 62, CYAN))


def run(text: str = None, audio_path: str = None, video_path: str = None):
    from inference_engine import predict

    transcript_text = None
    lang_code       = None

    if text:
        transcript_text = text

    elif audio_path:
        from media_processor import process_audio
        print(f"\n[CLI] Processing audio file: {audio_path}")
        try:
            transcript_text, lang_code = process_audio(audio_path)
        except (FileNotFoundError, ValueError, ImportError, RuntimeError) as e:
            print(f"\n  ✗ Audio processing failed: {e}")
            sys.exit(1)

    elif video_path:
        from media_processor import process_video
        print(f"\n[CLI] Processing video file: {video_path}")
        try:
            transcript_text, lang_code = process_video(video_path)
        except (FileNotFoundError, ValueError, EnvironmentError, RuntimeError) as e:
            print(f"\n  ✗ Video processing failed: {e}")
            sys.exit(1)

    print_header()

    try:
        result = predict(transcript_text)
    except Exception as e:
        print(f"\n  ✗ Inference failed: {e}")
        print("  → Make sure models are trained: python src/train_pipeline.py")
        sys.exit(1)

    print_result(result, transcript_text, lang_code)


def main():
    parser = argparse.ArgumentParser(
        description="Complaint Auto-Routing System — CLI",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python cli.py --text "Water pipe burst near the market"
          python cli.py --audio complaint.wav
          python cli.py --video complaint.mp4
        """)
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text",  type=str, help="Complaint text (any language)")
    group.add_argument("--audio", type=str, metavar="PATH",
                       help="Path to audio file (.mp3/.wav/.m4a/.ogg/.flac)")
    group.add_argument("--video", type=str, metavar="PATH",
                       help="Path to video file (.mp4/.avi/.mov/.mkv/.webm)")

    args = parser.parse_args()
    run(text=args.text, audio_path=args.audio, video_path=args.video)


if __name__ == "__main__":
    main()
