"""
AI Chat Agent with Voice Input
A modern chat application with text + audio input, powered by OpenRouter (GPT-5.2).
"""

import os
import sys
import tempfile
import threading
import json
import tkinter as tk
from tkinter import font as tkfont
from datetime import datetime

import requests
import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ─── Configuration ──────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAIRORUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CHAT_MODEL = "openai/gpt-5.2"
WHISPER_MODEL = "whisper-1"

# ─── Color Palette ──────────────────────────────────────────────────────────────
C = {
    "bg":              "#0D0D1A",
    "bg_chat":         "#111122",
    "bg_input":        "#1A1A30",
    "bg_user_bubble":  "#6C63FF",
    "bg_ai_bubble":    "#1E1E38",
    "bg_audio_bubble": "#3D1F6D",
    "accent":          "#6C63FF",
    "accent_hover":    "#5A52D5",
    "recording":       "#FF4757",
    "recording_hover": "#E8414F",
    "text_primary":    "#EAEAEA",
    "text_secondary":  "#9E9EB8",
    "text_muted":      "#5C5C72",
    "text_on_accent":  "#FFFFFF",
    "success":         "#2ED573",
    "warning":         "#FFA502",
    "border":          "#2A2A42",
    "scrollbar":       "#3A3A55",
}


# ─── Audio Functions ────────────────────────────────────────────────────────────

def record_audio(duration: float) -> np.ndarray:
    """Record audio from the default microphone."""
    return sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
    )


def wait_for_recording():
    """Block until the current recording finishes."""
    sd.wait()


def save_audio(audio_data: np.ndarray, filepath: str):
    """Save audio data to a WAV file."""
    sf.write(filepath, audio_data, SAMPLE_RATE)


# ─── OpenRouter API ─────────────────────────────────────────────────────────────

def chat_completion(messages: list[dict]) -> str:
    """Send messages to OpenRouter and get a response."""
    if not OPENROUTER_API_KEY:
        return "❌ Error: OPENROUTER_API_KEY not set in .env file."

    try:
        response = requests.post(
            url=f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "X-OpenRouter-Title": "Voice Chat Agent",
            },
            data=json.dumps({
                "model": CHAT_MODEL,
                "messages": messages,
            }),
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        error_body = ""
        try:
            error_body = e.response.json().get("error", {}).get("message", str(e))
        except Exception:
            error_body = str(e)
        return f"❌ API Error: {error_body}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def transcribe_audio_openrouter(filepath: str) -> str:
    """Transcribe audio using OpenRouter's Whisper-compatible endpoint."""
    if not OPENROUTER_API_KEY:
        return "❌ Error: OPENROUTER_API_KEY not set in .env file."

    # Use OpenAI-compatible whisper endpoint via OpenRouter
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
        with open(filepath, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=f,
                response_format="text",
            )
        return transcript
    except Exception:
        # Fallback: use OpenAI directly if the key is also an OpenAI key,
        # or return error
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                from openai import OpenAI as OAI
                client = OAI(api_key=openai_key)
                with open(filepath, "rb") as f:
                    transcript = client.audio.transcriptions.create(
                        model=WHISPER_MODEL,
                        file=f,
                        response_format="text",
                    )
                return transcript
        except Exception as e2:
            return f"[Transcription failed: {e2}]"
        return "[Transcription failed — check API key]"


# ─── Chat Bubble Widget ────────────────────────────────────────────────────────

class ChatBubble(tk.Frame):
    """A single chat message bubble."""

    def __init__(self, parent, sender: str, message: str, timestamp: str,
                 is_user: bool = True, is_audio: bool = False, **kwargs):
        super().__init__(parent, bg=C["bg_chat"], **kwargs)

        # Alignment
        anchor = tk.E if is_user else tk.W
        bubble_bg = C["bg_user_bubble"] if is_user else C["bg_ai_bubble"]
        if is_audio and is_user:
            bubble_bg = C["bg_audio_bubble"]
        text_color = C["text_on_accent"] if is_user else C["text_primary"]
        pad_side = (60, 8) if is_user else (8, 60)

        # Container for alignment
        container = tk.Frame(self, bg=C["bg_chat"])
        container.pack(fill=tk.X, padx=pad_side, pady=(4, 4))

        # Bubble frame
        bubble = tk.Frame(container, bg=bubble_bg)
        bubble.pack(anchor=anchor)

        # Audio indicator
        if is_audio and is_user:
            tk.Label(
                bubble, text="🎤 Voice message",
                font=tkfont.Font(family="Segoe UI", size=9),
                fg=C["text_secondary"], bg=bubble_bg,
            ).pack(anchor="w", padx=12, pady=(8, 0))

        # Message text
        msg_label = tk.Label(
            bubble, text=message,
            font=tkfont.Font(family="Segoe UI", size=11),
            fg=text_color, bg=bubble_bg,
            wraplength=280, justify=tk.LEFT,
            anchor="w",
        )
        msg_label.pack(padx=12, pady=(8, 4), anchor="w")

        # Timestamp
        tk.Label(
            bubble, text=timestamp,
            font=tkfont.Font(family="Segoe UI", size=8),
            fg=C["text_muted"] if not is_user else "#CCCCEE",
            bg=bubble_bg,
        ).pack(anchor="e", padx=12, pady=(0, 6))


# ─── Main Application ──────────────────────────────────────────────────────────

class ChatApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("💬 AI Voice Chat")
        self.root.configure(bg=C["bg"])
        self.root.resizable(True, True)

        # Window sizing
        w, h = 440, 780
        x = (root.winfo_screenwidth() - w) // 2
        y = (root.winfo_screenheight() - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.minsize(360, 600)

        # State
        self.is_recording = False
        self.is_processing = False
        self.duration = tk.DoubleVar(value=5.0)
        self.conversation: list[dict] = [
            {"role": "system", "content": (
                "You are a helpful, friendly AI assistant. "
                "You respond naturally and conversationally. "
                "Keep responses concise unless the user asks for detail."
            )}
        ]

        # Fonts
        self.f_title = tkfont.Font(family="Segoe UI", size=18, weight="bold")
        self.f_subtitle = tkfont.Font(family="Segoe UI", size=10)
        self.f_body = tkfont.Font(family="Segoe UI", size=12)
        self.f_small = tkfont.Font(family="Segoe UI", size=9)
        self.f_input = tkfont.Font(family="Segoe UI", size=12)
        self.f_btn = tkfont.Font(family="Segoe UI", size=12, weight="bold")
        self.f_label = tkfont.Font(family="Segoe UI", size=9, weight="bold")
        self.f_duration = tkfont.Font(family="Segoe UI", size=16, weight="bold")

        self._build_ui()
        self._add_welcome_message()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Header ──
        header = tk.Frame(self.root, bg=C["bg"], height=60)
        header.pack(fill=tk.X, padx=16, pady=(12, 0))
        header.pack_propagate(False)

        tk.Label(
            header, text="💬 AI Voice Chat",
            font=self.f_title, fg=C["text_primary"], bg=C["bg"]
        ).pack(side=tk.LEFT, pady=8)

        # Status dot
        self.status_frame = tk.Frame(header, bg=C["bg"])
        self.status_frame.pack(side=tk.RIGHT, pady=12)

        self.status_dot = tk.Label(
            self.status_frame, text="●",
            font=tkfont.Font(size=10), fg=C["success"], bg=C["bg"]
        )
        self.status_dot.pack(side=tk.LEFT, padx=(0, 4))

        self.status_text = tk.Label(
            self.status_frame, text="Online",
            font=self.f_small, fg=C["text_secondary"], bg=C["bg"]
        )
        self.status_text.pack(side=tk.LEFT)

        # Divider
        tk.Frame(self.root, bg=C["border"], height=1).pack(fill=tk.X, padx=16, pady=(4, 0))

        # ── Chat Area (scrollable) ──
        chat_container = tk.Frame(self.root, bg=C["bg_chat"])
        chat_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.chat_canvas = tk.Canvas(
            chat_container, bg=C["bg_chat"],
            highlightthickness=0, borderwidth=0,
        )
        self.scrollbar = tk.Scrollbar(
            chat_container, orient=tk.VERTICAL,
            command=self.chat_canvas.yview,
            bg=C["scrollbar"], troughcolor=C["bg_chat"],
            width=6, relief=tk.FLAT,
        )
        self.chat_frame = tk.Frame(self.chat_canvas, bg=C["bg_chat"])

        self.chat_frame.bind(
            "<Configure>",
            lambda e: self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))
        )

        self.canvas_window = self.chat_canvas.create_window(
            (0, 0), window=self.chat_frame, anchor="nw"
        )

        # Make the chat_frame expand to fill the canvas width
        self.chat_canvas.bind("<Configure>", self._on_canvas_configure)

        self.chat_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.chat_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Mouse wheel scrolling
        self.chat_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # ── Divider ──
        tk.Frame(self.root, bg=C["border"], height=1).pack(fill=tk.X, padx=16)

        # ── Audio Controls ──
        audio_bar = tk.Frame(self.root, bg=C["bg"], height=45)
        audio_bar.pack(fill=tk.X, padx=16, pady=(6, 2))

        # Duration display + slider
        dur_left = tk.Frame(audio_bar, bg=C["bg"])
        dur_left.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(
            dur_left, text="🎤",
            font=tkfont.Font(size=14), bg=C["bg"]
        ).pack(side=tk.LEFT, padx=(0, 6))

        self.dur_label = tk.Label(
            dur_left, text="5s",
            font=self.f_duration, fg=C["accent"], bg=C["bg"]
        )
        self.dur_label.pack(side=tk.LEFT, padx=(0, 8))

        self.slider = tk.Scale(
            dur_left, from_=1, to=30, orient=tk.HORIZONTAL,
            variable=self.duration, resolution=1,
            bg=C["bg"], fg=C["accent"],
            troughcolor=C["border"], activebackground=C["accent_hover"],
            highlightthickness=0, borderwidth=0,
            sliderrelief=tk.FLAT, length=120,
            showvalue=False,
            command=lambda v: self.dur_label.config(text=f"{int(float(v))}s")
        )
        self.slider.pack(side=tk.LEFT, padx=(0, 8))

        # Record button
        self.record_btn = tk.Button(
            audio_bar, text="● REC",
            font=self.f_btn, fg="#FFF", bg=C["recording"],
            activeforeground="#FFF", activebackground=C["recording_hover"],
            relief=tk.FLAT, cursor="hand2",
            padx=16, pady=4,
            command=self._on_record,
        )
        self.record_btn.pack(side=tk.RIGHT)
        self.record_btn.bind("<Enter>", lambda e: self.record_btn.config(
            bg=C["recording_hover"]) if not self.is_recording else None)
        self.record_btn.bind("<Leave>", lambda e: self.record_btn.config(
            bg=C["recording"]) if not self.is_recording else None)

        # ── Text Input Area ──
        input_bar = tk.Frame(self.root, bg=C["bg"], height=50)
        input_bar.pack(fill=tk.X, padx=16, pady=(4, 12))

        input_container = tk.Frame(input_bar, bg=C["bg_input"], highlightbackground=C["border"], highlightthickness=1)
        input_container.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=(0, 8))

        self.text_input = tk.Entry(
            input_container,
            font=self.f_input, fg=C["text_primary"], bg=C["bg_input"],
            insertbackground=C["text_primary"],
            relief=tk.FLAT, border=0,
        )
        self.text_input.pack(fill=tk.X, padx=12, pady=10)
        self.text_input.insert(0, "")
        self.text_input.bind("<Return>", self._on_send_text)

        # Placeholder
        self._set_placeholder()
        self.text_input.bind("<FocusIn>", self._on_focus_in)
        self.text_input.bind("<FocusOut>", self._on_focus_out)

        # Send button
        self.send_btn = tk.Button(
            input_bar, text="➤",
            font=tkfont.Font(size=16, weight="bold"),
            fg="#FFF", bg=C["accent"],
            activeforeground="#FFF", activebackground=C["accent_hover"],
            relief=tk.FLAT, cursor="hand2",
            padx=14, pady=4,
            command=self._on_send_text,
        )
        self.send_btn.pack(side=tk.RIGHT)
        self.send_btn.bind("<Enter>", lambda e: self.send_btn.config(bg=C["accent_hover"]))
        self.send_btn.bind("<Leave>", lambda e: self.send_btn.config(bg=C["accent"]))

    # ── Placeholder Logic ────────────────────────────────────────────────────

    def _set_placeholder(self):
        self.text_input.delete(0, tk.END)
        self.text_input.insert(0, "Type a message...")
        self.text_input.config(fg=C["text_muted"])
        self._placeholder_active = True

    def _on_focus_in(self, e=None):
        if getattr(self, '_placeholder_active', False):
            self.text_input.delete(0, tk.END)
            self.text_input.config(fg=C["text_primary"])
            self._placeholder_active = False

    def _on_focus_out(self, e=None):
        if not self.text_input.get().strip():
            self._set_placeholder()

    # ── Canvas / Scroll ──────────────────────────────────────────────────────

    def _on_canvas_configure(self, event):
        self.chat_canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        self.chat_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _scroll_to_bottom(self):
        self.chat_canvas.update_idletasks()
        self.chat_canvas.yview_moveto(1.0)

    # ── Add Messages ─────────────────────────────────────────────────────────

    def _add_welcome_message(self):
        """Show a welcome message from the AI."""
        bubble = ChatBubble(
            self.chat_frame,
            sender="AI",
            message="Hi! 👋 I'm your AI assistant. You can type a message or tap REC to send a voice message.",
            timestamp=datetime.now().strftime("%H:%M"),
            is_user=False,
        )
        bubble.pack(fill=tk.X)
        self._scroll_to_bottom()

    def _add_message(self, sender: str, message: str, is_user: bool, is_audio: bool = False):
        """Add a chat bubble to the conversation."""
        bubble = ChatBubble(
            self.chat_frame,
            sender=sender,
            message=message,
            timestamp=datetime.now().strftime("%H:%M"),
            is_user=is_user,
            is_audio=is_audio,
        )
        bubble.pack(fill=tk.X)
        self._scroll_to_bottom()

    def _add_typing_indicator(self):
        """Show a 'typing...' indicator."""
        self._typing_frame = tk.Frame(self.chat_frame, bg=C["bg_chat"])
        self._typing_frame.pack(fill=tk.X, padx=(8, 60), pady=(4, 4))

        bubble = tk.Frame(self._typing_frame, bg=C["bg_ai_bubble"])
        bubble.pack(anchor=tk.W)

        self._typing_label = tk.Label(
            bubble, text="● ● ●",
            font=tkfont.Font(family="Segoe UI", size=14),
            fg=C["text_muted"], bg=C["bg_ai_bubble"],
        )
        self._typing_label.pack(padx=16, pady=10)
        self._scroll_to_bottom()

        # Animate
        self._animate_typing()

    def _animate_typing(self):
        """Simple typing animation."""
        if not hasattr(self, '_typing_label') or not self._typing_label.winfo_exists():
            return
        current = self._typing_label.cget("text")
        dots = ["●      ●      ●", "  ●    ●    ●  ", "    ●  ●  ●    ", "  ●    ●    ●  "]
        try:
            idx = dots.index(current)
            next_text = dots[(idx + 1) % len(dots)]
        except ValueError:
            next_text = dots[0]
        self._typing_label.config(text=next_text)
        self._typing_anim_id = self.root.after(400, self._animate_typing)

    def _remove_typing_indicator(self):
        """Remove the typing indicator."""
        if hasattr(self, '_typing_anim_id'):
            self.root.after_cancel(self._typing_anim_id)
        if hasattr(self, '_typing_frame') and self._typing_frame.winfo_exists():
            self._typing_frame.destroy()

    # ── Set Status ───────────────────────────────────────────────────────────

    def _set_status(self, text: str, color: str):
        self.status_dot.config(fg=color)
        self.status_text.config(text=text)

    def _set_controls_enabled(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.record_btn.config(state=state)
        self.send_btn.config(state=state)
        self.text_input.config(state=state)
        self.slider.config(state=state)

    # ── Event Handlers ───────────────────────────────────────────────────────

    def _on_send_text(self, event=None):
        """Handle sending a text message."""
        if self.is_processing:
            return

        text = self.text_input.get().strip()
        if not text or getattr(self, '_placeholder_active', False):
            return

        # Clear input
        self.text_input.delete(0, tk.END)

        # Add user bubble
        self._add_message("You", text, is_user=True)

        # Add to conversation history
        self.conversation.append({"role": "user", "content": text})

        # Get AI response
        self._get_ai_response()

    def _on_record(self):
        """Handle record button click."""
        if self.is_processing:
            return

        if self.is_recording:
            return  # Already recording, wait for it to finish

        self.is_recording = True
        duration = self.duration.get()

        # Update UI
        self.record_btn.config(text="⏺ REC...", bg="#CC2233")
        self._set_status(f"Recording ({int(duration)}s)...", C["recording"])
        self._set_controls_enabled(False)
        self.record_btn.config(state=tk.NORMAL)  # Keep record btn visually active

        # Start recording in background
        thread = threading.Thread(
            target=self._record_and_process, args=(duration,), daemon=True
        )
        thread.start()

    # ── Background Tasks ─────────────────────────────────────────────────────

    def _record_and_process(self, duration: float):
        """Record audio, transcribe, and send to AI (background thread)."""
        tmp_path = None
        try:
            # Record
            audio_data = record_audio(duration)
            wait_for_recording()

            # Save
            tmp_path = tempfile.mktemp(suffix=".wav")
            save_audio(audio_data, tmp_path)

            # Update UI: transcribing
            self.root.after(0, self._update_ui_transcribing)

            # Transcribe
            transcribed_text = transcribe_audio_openrouter(tmp_path)

            if not transcribed_text or transcribed_text.startswith("[Transcription failed"):
                self.root.after(0, self._add_message, "You", f"🎤 (transcription failed)", True, True)
                self.root.after(0, self._reset_after_error)
                return

            # Show the transcribed message as user bubble
            self.root.after(0, self._add_message, "You", transcribed_text.strip(), True, True)

            # Add to conversation
            self.conversation.append({"role": "user", "content": transcribed_text.strip()})

            # Get AI response
            self.root.after(0, self._get_ai_response)

        except Exception as e:
            self.root.after(0, self._add_message, "System", f"❌ Error: {e}", False)
            self.root.after(0, self._reset_controls)
        finally:
            self.is_recording = False
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _get_ai_response(self):
        """Start getting AI response in background."""
        self.is_processing = True
        self._set_status("Thinking...", C["warning"])
        self._set_controls_enabled(False)
        self._add_typing_indicator()

        thread = threading.Thread(target=self._fetch_ai_response, daemon=True)
        thread.start()

    def _fetch_ai_response(self):
        """Call OpenRouter API (background thread)."""
        try:
            response_text = chat_completion(self.conversation)
            self.conversation.append({"role": "assistant", "content": response_text})
            self.root.after(0, self._show_ai_response, response_text)
        except Exception as e:
            self.root.after(0, self._show_ai_response, f"❌ Error: {e}")

    # ── UI Updates (main thread) ─────────────────────────────────────────────

    def _update_ui_transcribing(self):
        self._set_status("Transcribing...", C["warning"])
        self.record_btn.config(text="⏳ ...", bg=C["warning"])

    def _show_ai_response(self, text: str):
        self._remove_typing_indicator()
        self._add_message("AI", text, is_user=False)
        self._reset_controls()

    def _reset_after_error(self):
        self.is_processing = False
        self._reset_controls()

    def _reset_controls(self):
        self.is_processing = False
        self.is_recording = False
        self.record_btn.config(text="● REC", bg=C["recording"])
        self._set_status("Online", C["success"])
        self._set_controls_enabled(True)


# ─── Entry Point ────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
