"""
app.py
=======
Gradio web application for the Complaint Auto-Routing System.
Runs fully offline at http://localhost:7860

Features:
  - Three input tabs: Text | Audio | Video
  - Results: Officer + confidence, Priority badge, ETA, Top-5 similar, XAI explanation
  - Modern dark theme with custom CSS
"""

import os
import sys
import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── Lazy engine import (loads models on first call, not at import time) ──────
from inference_engine import predict as engine_predict
from media_processor  import process_audio, process_video, AUDIO_EXTENSIONS, VIDEO_EXTENSIONS

# ─────────────────────────────────────────────────────────────────────────────
# PRIORITY STYLING
# ─────────────────────────────────────────────────────────────────────────────
PRIORITY_EMOJI = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
PRIORITY_CSS   = {
    "High":   "background:#ff4444;color:#fff;padding:4px 14px;border-radius:20px;font-weight:700;",
    "Medium": "background:#ffaa00;color:#000;padding:4px 14px;border-radius:20px;font-weight:700;",
    "Low":    "background:#00cc66;color:#fff;padding:4px 14px;border-radius:20px;font-weight:700;",
}

OFFICER_ICONS = {
    "Water Officer":      "💧",
    "Electrical Officer": "⚡",
    "Road Officer":       "🚧",
    "Sanitation Officer": "🗑️",
    "Drainage Officer":   "🌊",
}


# ─────────────────────────────────────────────────────────────────────────────
# FORMAT RESULT → HTML for Gradio
# ─────────────────────────────────────────────────────────────────────────────
def format_results_html(result: dict, input_text: str, lang_code: str = None) -> str:
    officer  = result["officer"]
    conf     = result["officer_confidence"]
    priority = result["priority"]
    pri_conf = result["priority_confidence"]
    eta      = result["eta_days"]
    similar  = result["similar_complaints"]
    xai      = result["explanation"]
    all_prob = result.get("all_officer_probs", {})
    icon     = OFFICER_ICONS.get(officer, "🏛️")
    p_style  = PRIORITY_CSS.get(priority, "")
    p_emoji  = PRIORITY_EMOJI.get(priority, "")

    # ── Input preview ────────────────────────────────────────────────────────
    lang_note = f" <span style='color:#aaa;font-size:12px;'>(Detected: {lang_code})</span>" if lang_code else ""
    html = f"""
<div style='font-family:Inter,sans-serif;color:#e0e0e0;'>

  <div style='background:#1a1a2e;border:1px solid #2a2a4a;border-radius:12px;padding:16px;margin-bottom:16px;'>
    <div style='color:#7878cc;font-size:11px;font-weight:600;letter-spacing:1px;margin-bottom:6px;'>📝 COMPLAINT RECEIVED{lang_note}</div>
    <div style='color:#ddd;font-size:14px;line-height:1.5;font-style:italic;'>"{input_text[:300]}{"..." if len(input_text)>300 else ""}"</div>
  </div>

  <!-- Main result cards -->
  <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:16px;'>

    <div style='background:#0f3460;border-radius:12px;padding:16px;text-align:center;'>
      <div style='font-size:28px;margin-bottom:4px;'>{icon}</div>
      <div style='color:#7878cc;font-size:10px;font-weight:700;letter-spacing:1px;'>ASSIGNED OFFICER</div>
      <div style='color:#fff;font-size:15px;font-weight:700;margin-top:4px;'>{officer}</div>
      <div style='color:#aaa;font-size:12px;margin-top:4px;'>{conf:.1f}% confidence</div>
    </div>

    <div style='background:#1a1a2e;border:2px solid {"#ff4444" if priority=="High" else "#ffaa00" if priority=="Medium" else "#00cc66"};border-radius:12px;padding:16px;text-align:center;'>
      <div style='font-size:28px;margin-bottom:4px;'>{p_emoji}</div>
      <div style='color:#7878cc;font-size:10px;font-weight:700;letter-spacing:1px;'>PRIORITY</div>
      <div style='margin-top:6px;'><span style='{p_style}'>{priority}</span></div>
      <div style='color:#aaa;font-size:12px;margin-top:6px;'>{pri_conf:.1f}% confidence</div>
    </div>

    <div style='background:#0d2137;border-radius:12px;padding:16px;text-align:center;'>
      <div style='font-size:28px;margin-bottom:4px;'>⏱️</div>
      <div style='color:#7878cc;font-size:10px;font-weight:700;letter-spacing:1px;'>ESTIMATED RESOLUTION</div>
      <div style='color:#fff;font-size:22px;font-weight:700;margin-top:4px;'>{eta}</div>
      <div style='color:#aaa;font-size:12px;'>day{"s" if eta != 1 else ""}</div>
    </div>

  </div>
"""

    # ── Officer probability bars ──────────────────────────────────────────────
    html += """
  <div style='background:#12122a;border-radius:12px;padding:16px;margin-bottom:16px;'>
    <div style='color:#7878cc;font-size:11px;font-weight:700;letter-spacing:1px;margin-bottom:12px;'>📊 OFFICER CONFIDENCE BREAKDOWN</div>
"""
    for off, prob in sorted(all_prob.items(), key=lambda x: x[1], reverse=True):
        off_icon = OFFICER_ICONS.get(off, "🏛")
        is_top   = (off == officer)
        bar_color = "#4f8ef7" if is_top else "#2a3a5a"
        border    = "border:1px solid #4f8ef7;" if is_top else ""
        html += f"""
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:8px;{border}border-radius:8px;padding:4px 8px;'>
      <span style='width:24px;text-align:center;'>{off_icon}</span>
      <span style='width:150px;color:{"#fff" if is_top else "#aaa"};font-size:13px;{"font-weight:700;" if is_top else ""}'>{off}</span>
      <div style='flex:1;background:#1a1a2e;border-radius:4px;height:8px;'>
        <div style='width:{prob:.1f}%;background:{bar_color};border-radius:4px;height:8px;'></div>
      </div>
      <span style='width:48px;text-align:right;color:{"#4f8ef7" if is_top else "#666"};font-size:12px;font-weight:600;'>{prob:.1f}%</span>
    </div>
"""
    html += "  </div>\n"

    # ── Similar complaints ────────────────────────────────────────────────────
    html += """
  <div style='background:#12122a;border-radius:12px;padding:16px;margin-bottom:16px;'>
    <div style='color:#7878cc;font-size:11px;font-weight:700;letter-spacing:1px;margin-bottom:12px;'>🔍 TOP 5 SIMILAR PAST COMPLAINTS</div>
"""
    for i, s in enumerate(similar[:5], 1):
        sim_text  = s["text"][:100] + "…" if len(s["text"]) > 100 else s["text"]
        s_priority = s["priority"]
        s_style    = PRIORITY_CSS.get(s_priority, "")
        s_icon     = OFFICER_ICONS.get(s["officer"], "🏛")
        sim_pct    = s["similarity"]
        sim_color  = "#4f8ef7" if sim_pct >= 80 else "#7878cc" if sim_pct >= 60 else "#444"
        html += f"""
    <div style='border-bottom:1px solid #1e1e3a;padding:10px 0;{"padding-top:0;" if i==1 else ""}'>
      <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:4px;'>
        <span style='color:#bbb;font-size:13px;line-height:1.4;flex:1;margin-right:12px;'>{i}. {sim_text}</span>
        <span style='color:{sim_color};font-size:12px;font-weight:700;white-space:nowrap;'>{sim_pct:.1f}%</span>
      </div>
      <div style='display:flex;gap:8px;align-items:center;'>
        <span style='color:#8888aa;font-size:11px;'>{s_icon} {s["officer"]}</span>
        <span style='{s_style}font-size:10px;padding:2px 8px;border-radius:10px;'>{s_priority}</span>
        <span style='color:#666;font-size:11px;'>⏱ {s["eta_days"]} days</span>
      </div>
    </div>
"""
    html += "  </div>\n"

    # ── XAI Explanation ───────────────────────────────────────────────────────
    xai_clean = xai.replace("**", "").replace("*", "")
    xai_lines = xai_clean.splitlines()
    html += """
  <div style='background:#0a1628;border:1px solid #1e3a5a;border-radius:12px;padding:16px;'>
    <div style='color:#7878cc;font-size:11px;font-weight:700;letter-spacing:1px;margin-bottom:10px;'>🧠 AI EXPLANATION</div>
"""
    for line in xai_lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("🎯") or line.startswith("Why"):
            html += f"    <div style='color:#4f8ef7;font-weight:600;margin-bottom:6px;'>{line}</div>\n"
        elif line.startswith("•"):
            html += f"    <div style='color:#ccc;font-size:13px;margin:4px 0;padding-left:12px;'>{line}</div>\n"
        else:
            html += f"    <div style='color:#aaa;font-size:13px;margin:4px 0;'>{line}</div>\n"
    html += "  </div>\n</div>\n"

    return html


def format_error_html(msg: str) -> str:
    return f"""
<div style='font-family:Inter,sans-serif;background:#2a0f0f;border:1px solid #cc3333;
border-radius:12px;padding:20px;margin:10px 0;color:#ff9999;'>
  <div style='font-size:16px;font-weight:700;margin-bottom:8px;'>⚠️ Error</div>
  <div style='font-size:14px;'>{msg}</div>
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO HANDLER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def handle_text(complaint_text: str) -> str:
    if not complaint_text or not complaint_text.strip():
        return format_error_html("Please enter a complaint description.")
    try:
        result = engine_predict(complaint_text.strip())
        return format_results_html(result, complaint_text.strip())
    except FileNotFoundError:
        return format_error_html(
            "Models not found. Please run the training pipeline first:<br>"
            "<code>python data/data_generator.py</code><br>"
            "<code>python src/train_pipeline.py</code>"
        )
    except Exception as e:
        return format_error_html(f"Prediction error: {str(e)}")


def handle_audio(audio_file) -> str:
    if audio_file is None:
        return format_error_html("Please upload an audio file.")
    try:
        transcript, lang = process_audio(audio_file)
        result = engine_predict(transcript)
        return format_results_html(result, transcript, lang_code=lang)
    except EnvironmentError as e:
        return format_error_html(str(e))
    except FileNotFoundError as e:
        return format_error_html(str(e))
    except ImportError as e:
        return format_error_html(
            f"faster-whisper not installed: {e}<br>"
            "Run: <code>pip install faster-whisper</code>"
        )
    except Exception as e:
        return format_error_html(f"Audio processing error: {str(e)}")


def handle_video(video_file) -> str:
    if video_file is None:
        return format_error_html("Please upload a video file.")
    try:
        transcript, lang = process_video(video_file)
        result = engine_predict(transcript)
        return format_results_html(result, transcript, lang_code=lang)
    except EnvironmentError as e:
        return format_error_html(str(e))
    except FileNotFoundError as e:
        return format_error_html(str(e))
    except RuntimeError as e:
        return format_error_html(f"Video processing error: {str(e)}")
    except Exception as e:
        return format_error_html(f"Error: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: #0d0d1a !important;
}

.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
}

/* Tab styling */
.tab-nav button {
    background: #12122a !important;
    color: #8888cc !important;
    border: 1px solid #2a2a4a !important;
    border-radius: 8px 8px 0 0 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.tab-nav button.selected {
    background: #1e1e50 !important;
    color: #fff !important;
    border-bottom: 2px solid #4f8ef7 !important;
}
.tab-nav button:hover {
    color: #fff !important;
    background: #1a1a40 !important;
}

/* Inputs */
textarea, input[type="text"] {
    background: #12122a !important;
    color: #e0e0e0 !important;
    border: 1px solid #2a2a4a !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #4f8ef7 !important;
    box-shadow: 0 0 0 2px rgba(79,142,247,0.15) !important;
}

/* Buttons */
button.primary, .primary {
    background: linear-gradient(135deg, #4f8ef7, #7c3aed) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(79,142,247,0.3) !important;
}
button.primary:hover, .primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(79,142,247,0.45) !important;
}

/* Upload areas */
.upload-container, .file-preview {
    background: #12122a !important;
    border: 2px dashed #2a2a4a !important;
    border-radius: 12px !important;
    color: #8888cc !important;
}

/* Labels */
label span {
    color: #8888cc !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    letter-spacing: 0.5px !important;
}

/* Output HTML box */
.output-html {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d0d1a; }
::-webkit-scrollbar-thumb { background: #2a2a4a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #4f8ef7; }
"""


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO UI LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
EXAMPLE_COMPLAINTS = [
    "Water main broke on Elm Street causing the entire block to lose water supply since yesterday morning.",
    "The street light at the intersection of 5th and Broadway has been out for over a week, making it very unsafe at night.",
    "Large pothole near the school crossing on Oak Avenue is getting bigger and cars are swerving dangerously.",
    "Garbage has not been collected from our residential block in Sector 7 for the past 10 days, causing a serious health hazard.",
    "The catch basin on Park Lane is completely clogged. After last night's rain, the entire street was flooded knee-deep.",
    "Transformer on Cedar Road is sparking and making loud noises. Several houses lost power around 9 PM.",
]

def build_ui():
    with gr.Blocks(
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CUSTOM_CSS,
        title="Complaint Auto-Routing System",
    ) as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div style='text-align:center;padding:28px 0 12px;font-family:Inter,sans-serif;'>
          <div style='font-size:36px;margin-bottom:6px;'>🏛️</div>
          <h1 style='color:#fff;font-size:26px;font-weight:700;margin:0;
                     background:linear-gradient(90deg,#4f8ef7,#a78bfa);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            Municipal Complaint Auto-Routing System
          </h1>
          <p style='color:#7878cc;font-size:14px;margin-top:8px;'>
            AI-powered • Fully offline • Multilingual • Explainable
          </p>
          <div style='display:flex;justify-content:center;gap:12px;margin-top:12px;flex-wrap:wrap;'>
            <span style='background:#1e1e50;color:#aaa;padding:4px 12px;border-radius:20px;font-size:12px;'>💧 Water Officer</span>
            <span style='background:#1e1e50;color:#aaa;padding:4px 12px;border-radius:20px;font-size:12px;'>⚡ Electrical Officer</span>
            <span style='background:#1e1e50;color:#aaa;padding:4px 12px;border-radius:20px;font-size:12px;'>🚧 Road Officer</span>
            <span style='background:#1e1e50;color:#aaa;padding:4px 12px;border-radius:20px;font-size:12px;'>🗑️ Sanitation Officer</span>
            <span style='background:#1e1e50;color:#aaa;padding:4px 12px;border-radius:20px;font-size:12px;'>🌊 Drainage Officer</span>
          </div>
        </div>
        """)

        # ── Input Tabs ────────────────────────────────────────────────────────
        with gr.Tabs():

            # ── TEXT TAB ─────────────────────────────────────────────────────
            with gr.Tab("📝  Text Complaint"):
                with gr.Row():
                    with gr.Column(scale=3):
                        text_input = gr.Textbox(
                            label="Describe your complaint",
                            placeholder="e.g. The water pipe burst near the market and the road is flooded...",
                            lines=4,
                            max_lines=10,
                        )
                        with gr.Row():
                            text_btn   = gr.Button("🔍  Analyze Complaint", variant="primary", scale=2)
                            text_clear = gr.Button("🗑  Clear", scale=1)

                        gr.Markdown(
                            "**Examples:** (click to fill)",
                            elem_classes=["example-label"]
                        )
                        gr.Examples(
                            examples=EXAMPLE_COMPLAINTS,
                            inputs=text_input,
                            label="",
                        )

                text_output = gr.HTML(label="")
                text_btn.click(fn=handle_text,   inputs=text_input, outputs=text_output)
                text_clear.click(fn=lambda: ("", ""), outputs=[text_input, text_output])

            # ── AUDIO TAB ────────────────────────────────────────────────────
            with gr.Tab("🎙️  Audio Complaint"):
                gr.Markdown("""
Submit a voice recording of your complaint.
**Supported:** `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`
*Audio is transcribed locally using faster-whisper — no data sent anywhere.*
""")
                audio_input = gr.Audio(
                    label="Upload Audio File",
                    type="filepath",
                    sources=["upload"],
                )
                audio_btn    = gr.Button("🔍  Transcribe & Analyze", variant="primary")
                audio_output = gr.HTML(label="")

                audio_btn.click(fn=handle_audio, inputs=audio_input, outputs=audio_output)

            # ── VIDEO TAB ────────────────────────────────────────────────────
            with gr.Tab("🎥  Video Complaint"):
                gr.Markdown("""
Submit a video recording of your complaint.
**Supported:** `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`
*Audio is extracted via ffmpeg and transcribed locally — no data sent anywhere.*
> **Requires:** `ffmpeg` installed on your system PATH.
""")
                video_input = gr.Video(
                    label="Upload Video File",
                    sources=["upload"],
                )
                video_btn    = gr.Button("🔍  Extract & Analyze", variant="primary")
                video_output = gr.HTML(label="")

                video_btn.click(fn=handle_video, inputs=video_input, outputs=video_output)

        # ── Footer ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div style='text-align:center;color:#333;font-size:11px;margin-top:24px;font-family:Inter,sans-serif;'>
          Runs 100% offline &nbsp;|&nbsp; No data leaves your device &nbsp;|&nbsp;
          Models: paraphrase-multilingual-mpnet-base-v2 &nbsp;|&nbsp; faster-whisper &nbsp;|&nbsp; FAISS
        </div>
        """)

    return demo


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Complaint Auto-Routing System — Web App")
    parser.add_argument("--port",   type=int, default=7860, help="Port (default 7860)")
    parser.add_argument("--share",  action="store_true",    help="Create public Gradio share link")
    parser.add_argument("--host",   type=str, default="127.0.0.1")
    args = parser.parse_args()

    print("\n" + "=" * 58)
    print("  🏛  Complaint Auto-Routing System")
    print("  Local URL : http://127.0.0.1:7860")
    print("=" * 58 + "\n")

    app = build_ui()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )
