"""
Audio Transcription App
A modern GUI application that records audio and transcribes it using OpenAI Whisper.
Built with Tkinter for cross-platform compatibility.
"""

import os
import sys
import tempfile
import threading
import tkinter as tk
from tkinter import font as tkfont
from tkinter import filedialog

import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ─── Configuration ──────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"

# ─── Color Palette ──────────────────────────────────────────────────────────────
COLORS = {
    "bg_dark": "#0F0F1A",
    "bg_card": "#1A1A2E",
    "bg_input": "#16213E",
    "accent": "#6C63FF",
    "accent_hover": "#5A52D5",
    "accent_recording": "#FF4757",
    "accent_recording_hover": "#E8414F",
    "accent_transcribing": "#FFA502",
    "text_primary": "#EAEAEA",
    "text_secondary": "#8B8B9E",
    "text_muted": "#5C5C72",
    "success": "#2ED573",
    "border": "#2A2A40",
    "slider_trough": "#2A2A40",
    "slider_fill": "#6C63FF",
}


# ─── Audio Functions ────────────────────────────────────────────────────────────

def record_audio(duration: float) -> np.ndarray:
    """Record audio from the default microphone."""
    audio_data = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
    )
    sd.wait()
    return audio_data


def save_audio(audio_data: np.ndarray, filepath: str):
    """Save audio data to a WAV file."""
    sf.write(filepath, audio_data, SAMPLE_RATE)


def transcribe_audio(filepath: str, language: str = None) -> str:
    """Transcribe audio using OpenAI Whisper API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not set. Add it to your .env file."
        )

    client = OpenAI(api_key=api_key)

    with open(filepath, "rb") as audio_file:
        kwargs = {
            "model": "whisper-1",
            "file": audio_file,
            "response_format": "text",
        }
        if language:
            kwargs["language"] = language
        transcript = client.audio.transcriptions.create(**kwargs)

    return transcript


# ─── Rounded Rectangle Helper ──────────────────────────────────────────────────

def round_rect(canvas, x1, y1, x2, y2, radius=20, **kwargs):
    """Draw a rounded rectangle on a canvas."""
    points = [
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1,
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


# ─── Main Application ──────────────────────────────────────────────────────────

class TranscriptionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🎤 Voice Transcriber")
        self.root.configure(bg=COLORS["bg_dark"])
        self.root.resizable(True, True)

        # Window sizing — mobile-like proportions
        window_width = 420
        window_height = 750
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        x = (screen_w - window_width) // 2
        y = (screen_h - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.minsize(360, 600)

        # State
        self.is_recording = False
        self.is_transcribing = False
        self.duration = tk.DoubleVar(value=5.0)
        self.transcription_history: list[dict] = []

        # Fonts
        self.font_title = tkfont.Font(family="Segoe UI", size=22, weight="bold")
        self.font_subtitle = tkfont.Font(family="Segoe UI", size=11)
        self.font_body = tkfont.Font(family="Segoe UI", size=12)
        self.font_small = tkfont.Font(family="Segoe UI", size=10)
        self.font_button = tkfont.Font(family="Segoe UI", size=14, weight="bold")
        self.font_status = tkfont.Font(family="Segoe UI", size=12, weight="bold")
        self.font_duration = tkfont.Font(family="Segoe UI", size=28, weight="bold")
        self.font_label = tkfont.Font(family="Segoe UI", size=10, weight="bold")

        self._build_ui()

    # ── UI Construction ─────────────────────────────────────────────────────

    def _build_ui(self):
        """Build all UI components."""
        # Main scrollable container
        main_frame = tk.Frame(self.root, bg=COLORS["bg_dark"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # ── Header ──
        header = tk.Frame(main_frame, bg=COLORS["bg_dark"])
        header.pack(fill=tk.X, pady=(10, 5))

        tk.Label(
            header, text="🎤 Voice Transcriber",
            font=self.font_title, fg=COLORS["text_primary"],
            bg=COLORS["bg_dark"], anchor="w"
        ).pack(side=tk.LEFT)

        tk.Label(
            header, text="AI-Powered",
            font=self.font_small, fg=COLORS["accent"],
            bg=COLORS["bg_dark"], anchor="e"
        ).pack(side=tk.RIGHT, pady=(8, 0))

        # Divider
        tk.Frame(main_frame, bg=COLORS["border"], height=1).pack(fill=tk.X, pady=(5, 15))

        # ── Status Card ──
        self.status_card = tk.Frame(main_frame, bg=COLORS["bg_card"], highlightbackground=COLORS["border"], highlightthickness=1)
        self.status_card.pack(fill=tk.X, pady=(0, 15), ipady=12)

        self.status_icon = tk.Label(
            self.status_card, text="⏸",
            font=tkfont.Font(size=18), fg=COLORS["text_muted"],
            bg=COLORS["bg_card"]
        )
        self.status_icon.pack(pady=(8, 0))

        self.status_label = tk.Label(
            self.status_card, text="Ready to record",
            font=self.font_status, fg=COLORS["text_secondary"],
            bg=COLORS["bg_card"]
        )
        self.status_label.pack(pady=(2, 8))

        # ── Duration Control ──
        dur_frame = tk.Frame(main_frame, bg=COLORS["bg_card"], highlightbackground=COLORS["border"], highlightthickness=1)
        dur_frame.pack(fill=tk.X, pady=(0, 15), ipady=10)

        tk.Label(
            dur_frame, text="DURATION",
            font=self.font_label, fg=COLORS["text_muted"],
            bg=COLORS["bg_card"]
        ).pack(pady=(10, 0))

        self.duration_display = tk.Label(
            dur_frame, text="5s",
            font=self.font_duration, fg=COLORS["accent"],
            bg=COLORS["bg_card"]
        )
        self.duration_display.pack(pady=(0, 5))

        # Slider
        slider_frame = tk.Frame(dur_frame, bg=COLORS["bg_card"])
        slider_frame.pack(fill=tk.X, padx=25, pady=(0, 5))

        tk.Label(slider_frame, text="1s", font=self.font_small, fg=COLORS["text_muted"], bg=COLORS["bg_card"]).pack(side=tk.LEFT)
        tk.Label(slider_frame, text="30s", font=self.font_small, fg=COLORS["text_muted"], bg=COLORS["bg_card"]).pack(side=tk.RIGHT)

        self.slider = tk.Scale(
            dur_frame, from_=1, to=30, orient=tk.HORIZONTAL,
            variable=self.duration, resolution=1,
            bg=COLORS["bg_card"], fg=COLORS["accent"],
            troughcolor=COLORS["slider_trough"],
            activebackground=COLORS["accent_hover"],
            highlightthickness=0, borderwidth=0,
            sliderrelief=tk.FLAT, length=300,
            showvalue=False,
            command=self._on_duration_change
        )
        self.slider.pack(padx=25, pady=(0, 10))

        # ── Record Button ──
        self.btn_frame = tk.Frame(main_frame, bg=COLORS["bg_dark"])
        self.btn_frame.pack(fill=tk.X, pady=(0, 10))

        self.record_btn = tk.Button(
            self.btn_frame, text="● RECORD",
            font=self.font_button,
            fg="#FFFFFF", bg=COLORS["accent"],
            activeforeground="#FFFFFF", activebackground=COLORS["accent_hover"],
            relief=tk.FLAT, cursor="hand2",
            padx=20, pady=14,
            command=self._on_record_click
        )
        self.record_btn.pack(fill=tk.X, ipady=4)

        # Hover effects
        self.record_btn.bind("<Enter>", lambda e: self.record_btn.config(bg=COLORS["accent_hover"]) if not self.is_recording else None)
        self.record_btn.bind("<Leave>", lambda e: self._reset_button_color())

        # ── File transcribe button ──
        self.file_btn = tk.Button(
            self.btn_frame, text="📁 Transcribe File",
            font=self.font_small,
            fg=COLORS["text_secondary"], bg=COLORS["bg_card"],
            activeforeground=COLORS["text_primary"], activebackground=COLORS["bg_input"],
            relief=tk.FLAT, cursor="hand2",
            padx=10, pady=8,
            command=self._on_file_click
        )
        self.file_btn.pack(fill=tk.X, pady=(8, 0), ipady=2)

        # ── Transcription Output ──
        output_header = tk.Frame(main_frame, bg=COLORS["bg_dark"])
        output_header.pack(fill=tk.X, pady=(15, 5))

        tk.Label(
            output_header, text="TRANSCRIPTION",
            font=self.font_label, fg=COLORS["text_muted"],
            bg=COLORS["bg_dark"]
        ).pack(side=tk.LEFT)

        self.clear_btn = tk.Button(
            output_header, text="Clear",
            font=self.font_small, fg=COLORS["text_muted"],
            bg=COLORS["bg_dark"], relief=tk.FLAT,
            activeforeground=COLORS["text_primary"],
            activebackground=COLORS["bg_dark"],
            cursor="hand2", command=self._clear_output
        )
        self.clear_btn.pack(side=tk.RIGHT)

        # Text output area
        text_frame = tk.Frame(main_frame, bg=COLORS["border"], highlightthickness=0)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.output_text = tk.Text(
            text_frame,
            font=self.font_body,
            fg=COLORS["text_primary"], bg=COLORS["bg_card"],
            insertbackground=COLORS["text_primary"],
            selectbackground=COLORS["accent"],
            relief=tk.FLAT, wrap=tk.WORD,
            padx=15, pady=12,
            state=tk.DISABLED, cursor="arrow"
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        # Configure text tags
        self.output_text.tag_configure("timestamp", foreground=COLORS["text_muted"], font=self.font_small)
        self.output_text.tag_configure("transcription", foreground=COLORS["text_primary"], font=self.font_body)
        self.output_text.tag_configure("error", foreground=COLORS["accent_recording"], font=self.font_body)
        self.output_text.tag_configure("info", foreground=COLORS["accent"], font=self.font_small)
        self.output_text.tag_configure("divider", foreground=COLORS["border"], font=self.font_small)

        # ── Footer ──
        tk.Label(
            main_frame, text="Powered by OpenAI Whisper",
            font=self.font_small, fg=COLORS["text_muted"],
            bg=COLORS["bg_dark"]
        ).pack(pady=(0, 5))

    # ── Event Handlers ──────────────────────────────────────────────────────

    def _on_duration_change(self, value):
        """Update the duration display when slider changes."""
        self.duration_display.config(text=f"{int(float(value))}s")

    def _reset_button_color(self):
        """Reset the record button color based on state."""
        if self.is_recording:
            self.record_btn.config(bg=COLORS["accent_recording"])
        elif self.is_transcribing:
            self.record_btn.config(bg=COLORS["accent_transcribing"])
        else:
            self.record_btn.config(bg=COLORS["accent"])

    def _set_status(self, icon: str, text: str, color: str):
        """Update the status card."""
        self.status_icon.config(text=icon, fg=color)
        self.status_label.config(text=text, fg=color)

    def _append_output(self, text: str, tag: str = "transcription"):
        """Append text to the output area."""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, text, tag)
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)

    def _clear_output(self):
        """Clear the transcription output."""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.config(state=tk.DISABLED)
        self.transcription_history.clear()

    def _on_record_click(self):
        """Handle record button click."""
        if self.is_recording or self.is_transcribing:
            return

        self.is_recording = True
        duration = self.duration.get()

        # Update UI
        self.record_btn.config(text="● RECORDING...", bg=COLORS["accent_recording"])
        self._set_status("🔴", f"Recording ({int(duration)}s)...", COLORS["accent_recording"])
        self.slider.config(state=tk.DISABLED)
        self.file_btn.config(state=tk.DISABLED)

        # Start recording in background thread
        thread = threading.Thread(target=self._record_and_transcribe, args=(duration,), daemon=True)
        thread.start()

    def _on_file_click(self):
        """Handle file button click — open file dialog and transcribe."""
        if self.is_recording or self.is_transcribing:
            return

        filepath = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.m4a *.flac *.ogg *.webm"),
                ("All Files", "*.*"),
            ]
        )
        if not filepath:
            return

        self.is_transcribing = True
        self.record_btn.config(state=tk.DISABLED)
        self.file_btn.config(text="⏳ Transcribing...", state=tk.DISABLED)
        self._set_status("🔄", "Transcribing file...", COLORS["accent_transcribing"])

        thread = threading.Thread(target=self._transcribe_file, args=(filepath,), daemon=True)
        thread.start()

    # ── Background Tasks ────────────────────────────────────────────────────

    def _record_and_transcribe(self, duration: float):
        """Record audio, save, and transcribe (runs in background thread)."""
        try:
            # Record
            audio_data = record_audio(duration)

            # Save to temp file
            tmp_path = tempfile.mktemp(suffix=".wav")
            save_audio(audio_data, tmp_path)

            # Update UI for transcription phase
            self.root.after(0, self._update_ui_transcribing)

            # Transcribe
            text = transcribe_audio(tmp_path)

            # Show result
            self.root.after(0, self._show_transcription, text)

        except Exception as e:
            self.root.after(0, self._show_error, str(e))

        finally:
            # Cleanup
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            self.root.after(0, self._reset_ui)

    def _transcribe_file(self, filepath: str):
        """Transcribe an existing file (runs in background thread)."""
        try:
            text = transcribe_audio(filepath)
            filename = os.path.basename(filepath)
            self.root.after(0, self._show_transcription, text, filename)
        except Exception as e:
            self.root.after(0, self._show_error, str(e))
        finally:
            self.root.after(0, self._reset_ui)

    # ── UI Updates (main thread) ────────────────────────────────────────────

    def _update_ui_transcribing(self):
        """Switch UI to transcribing state."""
        self.is_recording = False
        self.is_transcribing = True
        self.record_btn.config(text="⏳ TRANSCRIBING...", bg=COLORS["accent_transcribing"])
        self._set_status("🔄", "Transcribing with Whisper...", COLORS["accent_transcribing"])

    def _show_transcription(self, text: str, source: str = "microphone"):
        """Display a transcription result."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")

        self._append_output(f"\n{'─' * 40}\n", "divider")
        self._append_output(f"  🕐 {timestamp}  •  📍 {source}\n", "timestamp")
        self._append_output(f"  {text.strip()}\n", "transcription")

        self.transcription_history.append({
            "time": timestamp,
            "source": source,
            "text": text.strip()
        })

    def _show_error(self, error: str):
        """Display an error in the output area."""
        self._append_output(f"\n{'─' * 40}\n", "divider")
        self._append_output(f"  ❌ Error: {error}\n", "error")

    def _reset_ui(self):
        """Reset UI to idle state."""
        self.is_recording = False
        self.is_transcribing = False
        self.record_btn.config(text="● RECORD", bg=COLORS["accent"], state=tk.NORMAL)
        self.file_btn.config(text="📁 Transcribe File", state=tk.NORMAL)
        self.slider.config(state=tk.NORMAL)
        self._set_status("⏸", "Ready to record", COLORS["text_secondary"])


# ─── Entry Point ────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()

    # Set app icon (if available)
    try:
        root.iconbitmap(default="")
    except Exception:
        pass

    app = TranscriptionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
