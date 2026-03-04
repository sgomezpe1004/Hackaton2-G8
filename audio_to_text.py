import threading, numpy as np, sounddevice as sd, tkinter as tk
import whisper

# Load Whisper model once at startup (use "base" for speed, "small" for accuracy)
print("Loading Whisper model (small)...")
model = whisper.load_model("small")
print("Model loaded!")

# ── Colors ──
C = dict(bg="#0D0D1A", panel="#111122", text="#EAEAEA", accent="#6C63FF",
         rec="#FF4757", warn="#FFA502", ok="#2ED573", dim="#9E9EB8", border="#2A2A42")

class App:
    def __init__(self, root):
        self.root = root
        root.title("🎙 Audio to Text")
        root.configure(bg=C["bg"])
        w, h = 500, 600
        root.geometry(f"{w}x{h}+{(root.winfo_screenwidth()-w)//2}+{(root.winfo_screenheight()-h)//2}")
        root.minsize(400, 400)
        self._recording = False
        self._frames = []

        # Header
        hdr = tk.Frame(root, bg=C["bg"])
        hdr.pack(fill=tk.X, padx=16, pady=(12, 0))
        tk.Label(hdr, text="🎙 Audio to Text", font=("Segoe UI", 20, "bold"),
                 fg=C["text"], bg=C["bg"]).pack(side=tk.LEFT)
        self.status = tk.Label(hdr, text="● Ready", font=("Segoe UI", 10),
                               fg=C["ok"], bg=C["bg"])
        self.status.pack(side=tk.RIGHT)
        tk.Frame(root, bg=C["border"], height=1).pack(fill=tk.X, padx=16, pady=8)

        # REC / STOP button
        btn_frame = tk.Frame(root, bg=C["bg"])
        btn_frame.pack(padx=16, pady=6)
        self.rec_btn = tk.Button(btn_frame, text="● REC", font=("Segoe UI", 14, "bold"),
                                 fg="#FFF", bg=C["rec"], relief=tk.FLAT, cursor="hand2",
                                 padx=30, pady=8, command=self.toggle_rec)
        self.rec_btn.pack()

        tk.Frame(root, bg=C["border"], height=1).pack(fill=tk.X, padx=16, pady=8)

        # Transcription output
        tk.Label(root, text="Transcription:", font=("Segoe UI", 11, "bold"),
                 fg=C["dim"], bg=C["bg"], anchor="w").pack(fill=tk.X, padx=20)
        txt_frame = tk.Frame(root, bg=C["border"], highlightthickness=1,
                             highlightbackground=C["border"])
        txt_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=(4, 8))
        self.output = tk.Text(txt_frame, font=("Segoe UI", 12), fg=C["text"],
                              bg=C["panel"], relief=tk.FLAT, wrap=tk.WORD,
                              insertbackground=C["text"], padx=12, pady=10)
        self.output.pack(fill=tk.BOTH, expand=True)

        # Clear button
        tk.Button(root, text="Clear", font=("Segoe UI", 10), fg=C["dim"],
                  bg=C["bg"], relief=tk.FLAT, cursor="hand2",
                  command=lambda: self.output.delete("1.0", tk.END)).pack(pady=(0, 10))

    def toggle_rec(self):
        if self._recording:
            self._recording = False
        else:
            self._recording = True
            self._frames = []
            self.rec_btn.config(text="■ STOP", bg=C["warn"])
            self.status.config(text="⏺ Recording...", fg=C["warn"])
            threading.Thread(target=self._record, daemon=True).start()

    def _record(self):
        try:
            def callback(indata, frames, time_info, status):
                if self._recording:
                    self._frames.append(indata.copy())

            with sd.InputStream(samplerate=16000, channels=1, dtype="float32", callback=callback):
                while self._recording:
                    sd.sleep(50)

            # Process recorded audio
            if not self._frames:
                self.root.after(0, self._reset, "Too short")
                return

            audio = np.concatenate(self._frames).flatten()
            if len(audio) < 1600:
                self.root.after(0, self._reset, "Too short")
                return

            self.root.after(0, lambda: (
                self.rec_btn.config(text="● REC", bg=C["rec"]),
                self.status.config(text="🔄 Transcribing...", fg=C["accent"])
            ))

            # Transcribe with Whisper (direct numpy, no ffmpeg needed)
            result = model.transcribe(audio, language="en", fp16=False,
                                      condition_on_previous_text=False,
                                      no_speech_threshold=0.5)
            text = result["text"].strip()

            self.root.after(0, lambda: self._show_result(text))

        except Exception as e:
            self.root.after(0, self._reset, f"Error: {e}")

    def _show_result(self, text):
        self.output.insert(tk.END, text + "\n\n")
        self.output.see(tk.END)
        self.status.config(text="● Ready", fg=C["ok"])

    def _reset(self, msg="● Ready"):
        self.rec_btn.config(text="● REC", bg=C["rec"])
        self.status.config(text=msg, fg=C["dim"] if "Error" in msg or "short" in msg else C["ok"])

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
