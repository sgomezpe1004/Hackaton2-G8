import os, json, tempfile, threading, tkinter as tk
from tkinter import font as tkfont
from datetime import datetime
import requests, numpy as np, sounddevice as sd, soundfile as sf
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = "openai/gpt-5.2"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Colors ──
C = dict(bg="#0D0D1A", chat="#111122", input="#1A1A30", user="#6C63FF",
         ai="#1E1E38", voice="#3D1F6D", accent="#6C63FF", rec="#FF4757",
         warn="#FFA502", text="#EAEAEA", dim="#9E9EB8", muted="#5C5C72",
         border="#2A2A42", ok="#2ED573")

def chat_api(messages):
    try:
        r = requests.post(API_URL, timeout=60,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            data=json.dumps({"model": MODEL, "messages": messages}))
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ {e}"

def transcribe(path):
    """Transcribe audio using Google's free speech recognition."""
    import speech_recognition as sr
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(path) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "[Could not understand audio]"
    except Exception as e:
        return f"[Transcription failed: {e}]"

class Bubble(tk.Frame):
    def __init__(self, parent, text, is_user=True, is_voice=False):
        super().__init__(parent, bg=C["chat"])
        bg = C["voice"] if is_voice else (C["user"] if is_user else C["ai"])
        pad = (60, 8) if is_user else (8, 60)
        box = tk.Frame(self, bg=C["chat"])
        box.pack(fill=tk.X, padx=pad, pady=3)
        bubble = tk.Frame(box, bg=bg)
        bubble.pack(anchor=tk.E if is_user else tk.W)
        if is_voice:
            tk.Label(bubble, text="🎤 Voice", font=("Segoe UI", 8), fg=C["dim"], bg=bg).pack(anchor="w", padx=10, pady=(6,0))
        tk.Label(bubble, text=text, font=("Segoe UI", 11), fg="#FFF" if is_user else C["text"],
                 bg=bg, wraplength=260, justify=tk.LEFT, anchor="w").pack(padx=10, pady=(6,4), anchor="w")
        tk.Label(bubble, text=datetime.now().strftime("%H:%M"), font=("Segoe UI", 8),
                 fg=C["muted"], bg=bg).pack(anchor="e", padx=10, pady=(0,5))

class App:
    def __init__(self, root):
        self.root = root
        root.title("💬 AI Voice Chat")
        root.configure(bg=C["bg"])
        w, h = 420, 720
        root.geometry(f"{w}x{h}+{(root.winfo_screenwidth()-w)//2}+{(root.winfo_screenheight()-h)//2}")
        root.minsize(360, 500)
        self.busy = False
        self.duration = tk.DoubleVar(value=5.0)
        self.history = [{"role": "system", "content": "You are a helpful AI assistant. Be concise."}]

        # Header
        hdr = tk.Frame(root, bg=C["bg"])
        hdr.pack(fill=tk.X, padx=16, pady=(10,0))
        tk.Label(hdr, text="💬 AI Voice Chat", font=("Segoe UI", 18, "bold"), fg=C["text"], bg=C["bg"]).pack(side=tk.LEFT)
        self.status = tk.Label(hdr, text="● Online", font=("Segoe UI", 9), fg=C["ok"], bg=C["bg"])
        self.status.pack(side=tk.RIGHT)
        tk.Frame(root, bg=C["border"], height=1).pack(fill=tk.X, padx=16, pady=4)

        # Chat area
        chat_box = tk.Frame(root, bg=C["chat"])
        chat_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=2)
        self.canvas = tk.Canvas(chat_box, bg=C["chat"], highlightthickness=0)
        self.chat = tk.Frame(self.canvas, bg=C["chat"])
        self.chat.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.win = self.canvas.create_window((0,0), window=self.chat, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self.win, width=e.width))
        self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(-int(e.delta/120), "units"))
        sb = tk.Scrollbar(chat_box, command=self.canvas.yview, width=5, bg=C["border"], troughcolor=C["chat"])
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        # Welcome
        Bubble(self.chat, "Hi! 👋 Type or tap REC to chat.", is_user=False).pack(fill=tk.X)

        tk.Frame(root, bg=C["border"], height=1).pack(fill=tk.X, padx=16)

        # Audio bar
        audio = tk.Frame(root, bg=C["bg"])
        audio.pack(fill=tk.X, padx=16, pady=4)
        self.dur_lbl = tk.Label(audio, text="5s", font=("Segoe UI", 14, "bold"), fg=C["accent"], bg=C["bg"])
        self.dur_lbl.pack(side=tk.LEFT, padx=(0,6))
        tk.Scale(audio, from_=1, to=30, orient=tk.HORIZONTAL, variable=self.duration, resolution=1,
                 bg=C["bg"], fg=C["accent"], troughcolor=C["border"], highlightthickness=0, borderwidth=0,
                 sliderrelief=tk.FLAT, length=140, showvalue=False,
                 command=lambda v: self.dur_lbl.config(text=f"{int(float(v))}s")).pack(side=tk.LEFT)
        self.rec_btn = tk.Button(audio, text="● REC", font=("Segoe UI", 11, "bold"), fg="#FFF",
                                 bg=C["rec"], relief=tk.FLAT, cursor="hand2", padx=14, command=self.on_rec)
        self.rec_btn.pack(side=tk.RIGHT)

        # Text input
        inp = tk.Frame(root, bg=C["bg"])
        inp.pack(fill=tk.X, padx=16, pady=(2,10))
        inp_box = tk.Frame(inp, bg=C["input"], highlightbackground=C["border"], highlightthickness=1)
        inp_box.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=(0,6))
        self.entry = tk.Entry(inp_box, font=("Segoe UI", 12), fg=C["text"], bg=C["input"],
                              insertbackground=C["text"], relief=tk.FLAT)
        self.entry.pack(fill=tk.X, padx=10, pady=8)
        self.entry.bind("<Return>", self.on_send)
        tk.Button(inp, text="➤", font=("Segoe UI", 14, "bold"), fg="#FFF", bg=C["accent"],
                  relief=tk.FLAT, cursor="hand2", padx=12, command=self.on_send).pack(side=tk.RIGHT)

    def add(self, text, is_user=True, is_voice=False):
        Bubble(self.chat, text, is_user, is_voice).pack(fill=tk.X)
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)

    def set_busy(self, busy, label="● Online"):
        self.busy = busy
        st = tk.DISABLED if busy else tk.NORMAL
        self.rec_btn.config(state=st)
        self.entry.config(state=st)
        self.status.config(text=label, fg=C["warn"] if busy else C["ok"])

    def on_send(self, e=None):
        text = self.entry.get().strip()
        if not text or self.busy: return
        self.entry.delete(0, tk.END)
        self.add(text, is_user=True)
        self.history.append({"role": "user", "content": text})
        self.ask_ai()

    def on_rec(self):
        if self.busy: return
        self.set_busy(True, "⏺ Recording...")
        self.rec_btn.config(text="⏺ ...", bg=C["warn"])
        threading.Thread(target=self._do_record, daemon=True).start()

    def _do_record(self):
        try:
            dur = self.duration.get()
            audio = sd.rec(int(dur * 16000), samplerate=16000, channels=1, dtype="float32")
            sd.wait()
            tmp = tempfile.mktemp(suffix=".wav")
            sf.write(tmp, audio, 16000)
            self.root.after(0, lambda: self.status.config(text="🔄 Transcribing...", fg=C["warn"]))
            text = transcribe(tmp)
            os.unlink(tmp)
            if not text or text.startswith("["):
                err = text or "No transcription returned"
                self.root.after(0, lambda: (self.add(f"🎤 {err}", True, True), self.set_busy(False)))
                return
            self.root.after(0, lambda: self.add(text.strip(), True, True))
            self.history.append({"role": "user", "content": text.strip()})
            self.root.after(0, self.ask_ai)
        except Exception as e:
            self.root.after(0, lambda: (self.add(f"❌ {e}", False), self.set_busy(False)))

    def ask_ai(self):
        self.set_busy(True, "💭 Thinking...")
        threading.Thread(target=self._do_ask, daemon=True).start()

    def _do_ask(self):
        reply = chat_api(self.history)
        self.history.append({"role": "assistant", "content": reply})
        self.root.after(0, lambda: (self.add(reply, False), self.set_busy(False),
                                     self.rec_btn.config(text="● REC", bg=C["rec"])))

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
 