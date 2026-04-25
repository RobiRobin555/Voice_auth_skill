"""
app/enrollment_gui.py — Giao diện đăng ký voiceprint (Tkinter, dark theme).

Cho phép người dùng ghi 5 mẫu giọng nói (~3 giây mỗi mẫu),
sau đó tính voiceprint embedding và lưu file .npy.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
import tkinter as tk
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# ── Colors (dark theme, đồng bộ với setup_wizard) ─────────────────────────

BG_PRIMARY = "#1a1a2e"
BG_SECONDARY = "#16213e"
BG_INPUT = "#0f3460"
FG_TEXT = "#e8e8e8"
FG_MUTED = "#8899aa"
FG_ACCENT = "#00d4ff"
FG_SUCCESS = "#00e676"
FG_ERROR = "#ff5252"
FG_RECORDING = "#ff6d00"
BTN_BG = "#0f3460"
BTN_SUCCESS = "#1b5e20"
FONT_TITLE = ("Segoe UI", 18, "bold")
FONT_LABEL = ("Segoe UI", 11)
FONT_BIG = ("Segoe UI", 14, "bold")
FONT_BUTTON = ("Segoe UI", 11, "bold")
FONT_SMALL = ("Segoe UI", 9)
FONT_MONO = ("Consolas", 10)

SAMPLE_RATE = 16000
RECORD_DURATION = 3.0    # giây mỗi mẫu
NUM_SAMPLES = 5


class EnrollmentGUI:
    """
    GUI đăng ký voiceprint.

    Parameters
    ----------
    config : ConfigManager
    on_complete : callable
        Gọi khi enrollment hoàn tất (voiceprint đã lưu).
    on_cancel : callable | None
        Gọi khi user huỷ.
    """

    def __init__(
        self,
        config,
        on_complete: Callable[[], None],
        on_cancel: Optional[Callable[[], None]] = None,
    ) -> None:
        self.config = config
        self.on_complete = on_complete
        self.on_cancel = on_cancel
        self._root: Optional[tk.Tk] = None

        self._recorded_files: list[str] = []
        self._current_sample = 0
        self._is_recording = False

    def show(self) -> None:
        """Hiển thị enrollment window."""
        root = tk.Tk()
        self._root = root
        root.title("Voice Auth — Đăng ký giọng nói")
        root.geometry("560x620")
        root.resizable(False, False)
        root.configure(bg=BG_PRIMARY)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        try:
            from app.config_manager import APP_ROOT
            ico = APP_ROOT / "assets" / "icon.ico"
            if ico.exists():
                root.iconbitmap(str(ico))
        except Exception:
            pass

        self._build_ui(root)
        root.mainloop()

    # ── Build UI ─────────────────────────────────────────────────────────

    def _build_ui(self, root: tk.Tk) -> None:
        # Header
        header = tk.Frame(root, bg=BG_SECONDARY, height=90)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header, text="🎙  Đăng ký giọng nói", font=FONT_TITLE,
            bg=BG_SECONDARY, fg=FG_ACCENT
        ).pack(pady=8)

        user = self.config.get("user", "user")
        tk.Label(
            header, text=f"Người dùng: {user}  •  {NUM_SAMPLES} mẫu x {RECORD_DURATION:.0f}s",
            font=FONT_SMALL, bg=BG_SECONDARY, fg=FG_MUTED
        ).pack()

        # ── Main content ─────────────────────────────────────────────────
        content = tk.Frame(root, bg=BG_PRIMARY, padx=40, pady=15)
        content.pack(fill="both", expand=True)

        # Instructions
        tk.Label(
            content,
            text="Nhấn nút bên dưới và đọc to bất kỳ câu nào trong 3 giây.\n"
                 "Ghi âm 5 mẫu để tạo dấu vân tay giọng nói chính xác.",
            font=FONT_LABEL, bg=BG_PRIMARY, fg=FG_TEXT,
            wraplength=460, justify="left"
        ).pack(fill="x", pady=(0, 15))

        # ── Sample progress ──────────────────────────────────────────────
        progress_frame = tk.Frame(content, bg=BG_PRIMARY)
        progress_frame.pack(fill="x", pady=(0, 10))

        self._sample_labels: list[tk.Label] = []
        for i in range(NUM_SAMPLES):
            lbl = tk.Label(
                progress_frame, text=f"  {i+1}  ", font=FONT_BIG,
                bg=BG_SECONDARY, fg=FG_MUTED, width=4,
                relief="flat", bd=0
            )
            lbl.pack(side="left", expand=True, fill="x", padx=3, ipady=8)
            self._sample_labels.append(lbl)

        # ── Waveform canvas ──────────────────────────────────────────────
        self._canvas = tk.Canvas(
            content, width=470, height=100,
            bg=BG_SECONDARY, highlightthickness=1, highlightbackground=BG_INPUT
        )
        self._canvas.pack(fill="x", pady=10)
        self._draw_empty_waveform()

        # ── Status ───────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Sẵn sàng ghi mẫu 1")
        self._status_label = tk.Label(
            content, textvariable=self._status_var, font=FONT_LABEL,
            bg=BG_PRIMARY, fg=FG_ACCENT, wraplength=460
        )
        self._status_label.pack(fill="x", pady=5)

        # ── Countdown ────────────────────────────────────────────────────
        self._countdown_var = tk.StringVar(value="")
        tk.Label(
            content, textvariable=self._countdown_var, font=("Segoe UI", 36, "bold"),
            bg=BG_PRIMARY, fg=FG_RECORDING
        ).pack(pady=5)

        # ── Buttons ──────────────────────────────────────────────────────
        btn_frame = tk.Frame(content, bg=BG_PRIMARY)
        btn_frame.pack(fill="x", pady=(10, 0))

        self._record_btn = tk.Button(
            btn_frame, text="🎙  Bắt đầu ghi âm", font=FONT_BUTTON,
            bg=BTN_BG, fg=FG_TEXT, activebackground="#1a5276",
            activeforeground=FG_ACCENT, relief="flat", bd=0,
            cursor="hand2", command=self._start_recording, padx=15, pady=12
        )
        self._record_btn.pack(side="left", expand=True, fill="x", padx=(0, 5))

        self._done_btn = tk.Button(
            btn_frame, text="✅  Hoàn tất", font=FONT_BUTTON,
            bg=BG_SECONDARY, fg=FG_MUTED, relief="flat", bd=0,
            state="disabled", padx=15, pady=12
        )
        self._done_btn.pack(side="right", expand=True, fill="x", padx=(5, 0))

    # ── Waveform drawing ─────────────────────────────────────────────────

    def _draw_empty_waveform(self) -> None:
        self._canvas.delete("all")
        w = int(self._canvas.cget("width"))
        h = int(self._canvas.cget("height"))
        mid = h // 2
        self._canvas.create_line(0, mid, w, mid, fill=FG_MUTED, dash=(4, 4))
        self._canvas.create_text(
            w // 2, mid, text="Waveform sẽ hiện ở đây",
            fill=FG_MUTED, font=FONT_SMALL
        )

    def _draw_waveform(self, audio) -> None:
        import numpy as np
        self._canvas.delete("all")
        w = int(self._canvas.cget("width"))
        h = int(self._canvas.cget("height"))
        mid = h // 2

        # Downsample to canvas width
        step = max(1, len(audio) // w)
        samples = audio[::step][:w]

        points = []
        for i, s in enumerate(samples):
            y = mid - int(s * mid * 0.9)
            y = max(2, min(h - 2, y))
            points.append((i, y))

        # Draw filled waveform
        for i, (x, y) in enumerate(points):
            color = FG_ACCENT if abs(y - mid) > 5 else FG_MUTED
            self._canvas.create_line(x, mid, x, y, fill=color, width=1)

    # ── Recording ────────────────────────────────────────────────────────

    def _start_recording(self) -> None:
        if self._is_recording:
            return
        if self._current_sample >= NUM_SAMPLES:
            return

        self._is_recording = True
        self._record_btn.config(state="disabled", text="⏺  Đang ghi...")

        # Countdown 3-2-1 rồi ghi
        self._countdown(3)

    def _countdown(self, n: int) -> None:
        if n > 0:
            self._countdown_var.set(str(n))
            self._set_status(f"Chuẩn bị... {n}", FG_RECORDING)
            self._root.after(1000, lambda: self._countdown(n - 1))
        else:
            self._countdown_var.set("🔴")
            self._set_status(f"🎙 ĐÃ GHI MẪU {self._current_sample + 1}/{NUM_SAMPLES} — Hãy nói!", FG_RECORDING)
            threading.Thread(target=self._record_audio, daemon=True).start()

    def _record_audio(self) -> None:
        try:
            import sounddevice as sd
        except ImportError:
            self._root.after(0, lambda: self._set_status(
                "❌ Thiếu sounddevice: pip install sounddevice", FG_ERROR
            ))
            self._is_recording = False
            return

        try:
            audio = sd.rec(
                int(RECORD_DURATION * SAMPLE_RATE),
                samplerate=SAMPLE_RATE, channels=1, dtype="float32"
            )
            sd.wait()
            audio = audio.flatten()

            # Save temp wav
            import soundfile as sf
            fd, path = tempfile.mkstemp(suffix=".wav", prefix="enroll_")
            os.close(fd)
            sf.write(path, audio, SAMPLE_RATE)
            self._recorded_files.append(path)

            self._root.after(0, lambda a=audio: self._on_recording_done(a))

        except Exception as exc:
            self._root.after(0, lambda: self._set_status(
                f"❌ Lỗi ghi âm: {exc}", FG_ERROR
            ))
            self._is_recording = False

    def _on_recording_done(self, audio) -> None:
        self._countdown_var.set("")
        self._is_recording = False

        # Update sample indicator
        lbl = self._sample_labels[self._current_sample]
        lbl.config(bg=FG_SUCCESS, fg=BG_PRIMARY)

        self._current_sample += 1

        # Draw waveform
        self._draw_waveform(audio)

        if self._current_sample >= NUM_SAMPLES:
            self._set_status("✅ Đã ghi đủ mẫu! Nhấn 'Hoàn tất' để lưu.", FG_SUCCESS)
            self._record_btn.config(state="disabled", text="🎙 Đã đủ mẫu")
            self._done_btn.config(
                state="normal", bg=BTN_SUCCESS, fg=FG_TEXT,
                activebackground="#2e7d32", cursor="hand2",
                command=self._compute_voiceprint
            )
        else:
            self._set_status(
                f"✅ Mẫu {self._current_sample} OK! Nhấn để ghi mẫu {self._current_sample + 1}",
                FG_SUCCESS
            )
            self._record_btn.config(
                state="normal",
                text=f"🎙  Ghi mẫu {self._current_sample + 1}"
            )

    # ── Compute voiceprint ───────────────────────────────────────────────

    def _compute_voiceprint(self) -> None:
        self._set_status("⏳ Đang tính voiceprint...", FG_ACCENT)
        self._done_btn.config(state="disabled")

        threading.Thread(target=self._do_compute, daemon=True).start()

    def _do_compute(self) -> None:
        try:
            import numpy as np
            from auth.verifier import VoiceVerifier

            model_path = self.config.model_path
            cohort_dir = self.config.cohort_dir
            voiceprint_path = self.config.voiceprint_path
            threshold = self.config.threshold
            log_path = self.config.log_path

            # Tạo verifier chỉ để dùng enroll (bỏ qua voiceprint load)
            # Cần sửa nhỏ: VoiceVerifier yêu cầu voiceprint tồn tại
            # → tạo dummy trước
            vp_dir = Path(voiceprint_path).parent
            vp_dir.mkdir(parents=True, exist_ok=True)

            # Tạo dummy voiceprint tạm để khởi tạo verifier
            dummy_vp = str(vp_dir / "__temp_enroll__.npy")
            np.save(dummy_vp, np.zeros(192, dtype=np.float32))

            try:
                verifier = VoiceVerifier(
                    model_path=model_path,
                    voiceprint_path=dummy_vp,
                    cohort_dir=cohort_dir,
                    threshold_asnorm=threshold,
                    log_path=log_path,
                )

                voiceprint = verifier.enroll(
                    audio_files=self._recorded_files,
                    save_path=voiceprint_path,
                )

                self.config.update(
                    voiceprint_path=voiceprint_path,
                    enrolled=True,
                )

                self._root.after(0, lambda: self._on_enroll_success(voiceprint_path))
            finally:
                # Xóa dummy
                try:
                    os.unlink(dummy_vp)
                except OSError:
                    pass

        except Exception as exc:
            logger.error(f"[Enrollment] Lỗi: {exc}", exc_info=True)
            self._root.after(0, lambda: self._set_status(
                f"❌ Lỗi tính voiceprint: {exc}", FG_ERROR
            ))
            self._root.after(0, lambda: self._done_btn.config(state="normal"))

    def _on_enroll_success(self, path: str) -> None:
        self._set_status(f"✅ Voiceprint đã lưu → {Path(path).name}", FG_SUCCESS)
        self._cleanup_temp_files()

        self._done_btn.config(
            text="🚀  Bắt đầu sử dụng",
            state="normal", bg=BTN_SUCCESS, fg=FG_TEXT,
            command=self._finish
        )

    # ── Cleanup / Finish ─────────────────────────────────────────────────

    def _cleanup_temp_files(self) -> None:
        for p in self._recorded_files:
            try:
                os.unlink(p)
            except OSError:
                pass
        self._recorded_files.clear()

    def _finish(self) -> None:
        self._cleanup_temp_files()
        if self._root:
            self._root.destroy()
            self._root = None
        self.on_complete()

    def _on_close(self) -> None:
        self._cleanup_temp_files()
        if self.on_cancel:
            self.on_cancel()
        if self._root:
            self._root.destroy()
            self._root = None

    def _set_status(self, text: str, color: str = FG_MUTED) -> None:
        self._status_var.set(text)
        self._status_label.config(fg=color)
