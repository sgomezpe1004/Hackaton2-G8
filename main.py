import os, json, base64, tempfile, threading, tkinter as tk
from tkinter import font as tkfont
from datetime import datetime
import requests, numpy as np, sounddevice as sd, soundfile as sf
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = "openai/gpt-5.2"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
BASE = os.path.dirname(os.path.abspath(__file__))

# ── Parse HTML schedules ──
def parse_schedules():
    """Extract shift data from HTML schedule files into plain text."""
    import re, html as htmlmod
    text = ""
    for fname in ("marchschedule.html", "aprilschedule.html"):
        path = os.path.join(BASE, fname)
        if not os.path.exists(path): continue
        with open(path, encoding="utf-8") as f: raw = f.read()
        month = fname.replace("schedule.html", "").upper()
        text += f"\n=== {month} 2026 SCHEDULE ===\n"
        tags = re.findall(r'class="(day-name|day-date|shift-station|shift-unit|shift-time|shift-medics)">(.*?)<', raw)
        station, unit, day_line = "", "", ""
        for cls, val in tags:
            val = htmlmod.unescape(val)
            if cls == "day-name": day_line = val
            elif cls == "day-date": text += f"\n{day_line} {val}:\n"; station = unit = ""
            elif cls == "shift-station": station = val
            elif cls == "shift-unit": unit = val
            elif cls == "shift-time":
                label = f"{station} | {unit}" if unit else station
                text += f"  {label} | {val} | "
            elif cls == "shift-medics":
                text += f"{val}\n"; unit = ""
    return text

SCHEDULES = parse_schedules()

SYSTEM = f"""You are an EAI Ambulance Service scheduling assistant. Today is {datetime.now().strftime('%B %d, %Y')}.

SCHEDULES (March & April 2026):
{SCHEDULES}

CAPABILITIES:
1. **Schedule Lookup** — When a user asks about a team's schedule (e.g. "What's Team01's schedule next week?"), find and list their shifts from the data above.
2. **Shift Change Request** — When a user wants a shift change, collect ALL required fields:
   - First Name
   - Last Name
   - Medic Number (e.g. Team07)
   - Shift Day (date)
   - Shift Start time
   - Shift End time
   - Requested Action: Day Off Request, Swap Shift, Vacation Day, or Other (if Other, ask for Reason)

   After each user message, list which fields are filled and which are MISSING.
   Once ALL fields are complete, display a summary like:
   ✅ SHIFT CHANGE REQUEST COMPLETE
   Name: [First] [Last]
   Medic Number: [number]
   Shift Day: [date]
   Shift Start: [time]
   Shift End: [time]
   Action: [action]
   (Reason: [reason] — only if Other)
   
   Then tell the user to submit via the Shift Change Request form.

Be concise. Use the schedule data to answer questions accurately."""

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
    """Transcribe audio using OpenRouter gpt-4o-audio-preview."""
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        r = requests.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}",
                          "Content-Type": "application/json"},
                          json={"model": "openai/gpt-4o-audio-preview",
                                "messages": [{"role": "user", "content": [
                                    {"type": "text", "text": "Transcribe this audio exactly. Return only the transcription, nothing else."},
                                    {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}]}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
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
        root.title("🚑 EAI Schedule Chat")
        root.configure(bg=C["bg"])
        w, h = 420, 720
        root.geometry(f"{w}x{h}+{(root.winfo_screenwidth()-w)//2}+{(root.winfo_screenheight()-h)//2}")
        root.minsize(360, 500)
        self.busy = False
        self.history = [{"role": "system", "content": SYSTEM}]

        # Header
        hdr = tk.Frame(root, bg=C["bg"])
        hdr.pack(fill=tk.X, padx=16, pady=(10,0))
        tk.Label(hdr, text="🚑 EAI Schedule Chat", font=("Segoe UI", 18, "bold"), fg=C["text"], bg=C["bg"]).pack(side=tk.LEFT)
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
        Bubble(self.chat, "Hi! 🚑 Ask about schedules or request a shift change.", is_user=False).pack(fill=tk.X)

        tk.Frame(root, bg=C["border"], height=1).pack(fill=tk.X, padx=16)

        # REC button
        rec_bar = tk.Frame(root, bg=C["bg"])
        rec_bar.pack(fill=tk.X, padx=16, pady=4)
        self.rec_btn = tk.Button(rec_bar, text="● REC", font=("Segoe UI", 11, "bold"), fg="#FFF",
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
            audio = sd.rec(int(10 * 16000), samplerate=16000, channels=1, dtype="float32")
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
 