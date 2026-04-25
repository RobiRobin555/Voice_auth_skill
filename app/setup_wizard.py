"""
app/setup_wizard.py — Setup Wizard GUI (Tkinter, dark theme).

Lần khởi động đầu tiên, người dùng cần cung cấp:
  1. OpenClaw Gateway URL
  2. Token (Bearer)
  3. Username

Có nút "Test kết nối" trước khi lưu.
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# ── Dark theme colors ──────────────────────────────────────────────────────

BG_PRIMARY = "#1a1a2e"
BG_SECONDARY = "#16213e"
BG_INPUT = "#0f3460"
FG_TEXT = "#e8e8e8"
FG_MUTED = "#8899aa"
FG_ACCENT = "#00d4ff"
FG_SUCCESS = "#00e676"
FG_ERROR = "#ff5252"
BTN_BG = "#0f3460"
BTN_HOVER = "#1a5276"
BTN_SUCCESS = "#1b5e20"
FONT_TITLE = ("Segoe UI", 18, "bold")
FONT_LABEL = ("Segoe UI", 11)
FONT_INPUT = ("Consolas", 11)
FONT_BUTTON = ("Segoe UI", 11, "bold")
FONT_SMALL = ("Segoe UI", 9)


class SetupWizard:
    """
    GUI cài đặt ban đầu: Gateway URL + Token + Username.

    Parameters
    ----------
    config : ConfigManager
        Config manager instance.
    on_complete : callable
        Gọi khi user hoàn tất setup (lưu settings thành công).
    on_cancel : callable | None
        Gọi khi user huỷ → thoát app.
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
        self._show_password = False

    def show(self) -> None:
        """Hiển thị wizard window."""
        root = tk.Tk()
        self._root = root
        root.title("Voice Auth — Thiết lập Gateway")
        root.geometry("520x580")
        root.resizable(False, False)
        root.configure(bg=BG_PRIMARY)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Đặt icon ─────────────────────────────────────────────────────
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
        # ── Header ───────────────────────────────────────────────────────
        header = tk.Frame(root, bg=BG_SECONDARY, height=80)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header, text="🎤  Voice Auth Tray", font=FONT_TITLE,
            bg=BG_SECONDARY, fg=FG_ACCENT
        ).pack(pady=10)

        tk.Label(
            header, text="Kết nối ứng dụng với OpenClaw Gateway",
            font=FONT_SMALL, bg=BG_SECONDARY, fg=FG_MUTED
        ).pack()

        # ── Form container ───────────────────────────────────────────────
        form = tk.Frame(root, bg=BG_PRIMARY, padx=40, pady=20)
        form.pack(fill="both", expand=True)

        # ── Gateway URL ──────────────────────────────────────────────────
        tk.Label(
            form, text="Gateway URL", font=FONT_LABEL,
            bg=BG_PRIMARY, fg=FG_TEXT, anchor="w"
        ).pack(fill="x", pady=(10, 2))

        self._url_var = tk.StringVar(value=self.config.get("gateway_url", "http://localhost:18789"))
        url_entry = tk.Entry(
            form, textvariable=self._url_var, font=FONT_INPUT,
            bg=BG_INPUT, fg=FG_TEXT, insertbackground=FG_ACCENT,
            relief="flat", bd=0, highlightthickness=2,
            highlightcolor=FG_ACCENT, highlightbackground=BG_SECONDARY
        )
        url_entry.pack(fill="x", ipady=8, pady=(0, 5))

        tk.Label(
            form, text="Mặc định: http://localhost:18789", font=FONT_SMALL,
            bg=BG_PRIMARY, fg=FG_MUTED, anchor="w"
        ).pack(fill="x")

        # ── Token ────────────────────────────────────────────────────────
        tk.Label(
            form, text="Token", font=FONT_LABEL,
            bg=BG_PRIMARY, fg=FG_TEXT, anchor="w"
        ).pack(fill="x", pady=(15, 2))

        token_frame = tk.Frame(form, bg=BG_PRIMARY)
        token_frame.pack(fill="x")

        self._token_var = tk.StringVar(value=self.config.get("gateway_token", ""))
        self._token_entry = tk.Entry(
            token_frame, textvariable=self._token_var, font=FONT_INPUT,
            bg=BG_INPUT, fg=FG_TEXT, insertbackground=FG_ACCENT,
            relief="flat", bd=0, show="●",
            highlightthickness=2, highlightcolor=FG_ACCENT,
            highlightbackground=BG_SECONDARY
        )
        self._token_entry.pack(side="left", fill="x", expand=True, ipady=8)

        show_btn = tk.Button(
            token_frame, text="👁", font=("Segoe UI", 12),
            bg=BG_INPUT, fg=FG_MUTED, relief="flat", bd=0,
            activebackground=BTN_HOVER, activeforeground=FG_ACCENT,
            cursor="hand2", command=self._toggle_password
        )
        show_btn.pack(side="right", padx=(5, 0), ipady=4, ipadx=5)

        tk.Label(
            form, text="Lấy token từ: openclaw config get gateway.token",
            font=FONT_SMALL, bg=BG_PRIMARY, fg=FG_MUTED, anchor="w"
        ).pack(fill="x")

        # ── Username ─────────────────────────────────────────────────────
        tk.Label(
            form, text="Tên người dùng", font=FONT_LABEL,
            bg=BG_PRIMARY, fg=FG_TEXT, anchor="w"
        ).pack(fill="x", pady=(15, 2))

        self._user_var = tk.StringVar(value=self.config.get("user", "user"))
        user_entry = tk.Entry(
            form, textvariable=self._user_var, font=FONT_INPUT,
            bg=BG_INPUT, fg=FG_TEXT, insertbackground=FG_ACCENT,
            relief="flat", bd=0, highlightthickness=2,
            highlightcolor=FG_ACCENT, highlightbackground=BG_SECONDARY
        )
        user_entry.pack(fill="x", ipady=8)

        # ── Status label ─────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="")
        self._status_label = tk.Label(
            form, textvariable=self._status_var, font=FONT_SMALL,
            bg=BG_PRIMARY, fg=FG_MUTED, anchor="w", wraplength=420
        )
        self._status_label.pack(fill="x", pady=(15, 5))

        # ── Buttons ──────────────────────────────────────────────────────
        btn_frame = tk.Frame(form, bg=BG_PRIMARY)
        btn_frame.pack(fill="x", pady=(10, 0))

        self._test_btn = tk.Button(
            btn_frame, text="🔌  Test kết nối", font=FONT_BUTTON,
            bg=BTN_BG, fg=FG_TEXT, activebackground=BTN_HOVER,
            activeforeground=FG_ACCENT, relief="flat", bd=0,
            cursor="hand2", command=self._test_connection, padx=15, pady=10
        )
        self._test_btn.pack(side="left", expand=True, fill="x", padx=(0, 5))

        self._save_btn = tk.Button(
            btn_frame, text="✅  Lưu & Tiếp tục", font=FONT_BUTTON,
            bg=BTN_SUCCESS, fg=FG_TEXT, activebackground="#2e7d32",
            activeforeground="#ffffff", relief="flat", bd=0,
            cursor="hand2", command=self._save_and_continue, padx=15, pady=10
        )
        self._save_btn.pack(side="right", expand=True, fill="x", padx=(5, 0))

    # ── Actions ──────────────────────────────────────────────────────────

    def _toggle_password(self) -> None:
        self._show_password = not self._show_password
        self._token_entry.config(show="" if self._show_password else "●")

    def _test_connection(self) -> None:
        """Test kết nối Gateway trong background thread."""
        url = self._url_var.get().strip()
        token = self._token_var.get().strip()

        if not url:
            self._set_status("⚠️  Nhập Gateway URL trước", FG_ERROR)
            return

        self._set_status("⏳  Đang kiểm tra kết nối...", FG_MUTED)
        self._test_btn.config(state="disabled")

        def _run():
            from app.openclaw_client import OpenClawClient
            client = OpenClawClient(gateway_url=url, token=token, timeout=10)
            ok, msg = client.test_connection()
            self._root.after(0, lambda: self._on_test_result(ok, msg))

        threading.Thread(target=_run, daemon=True).start()

    def _on_test_result(self, ok: bool, msg: str) -> None:
        self._test_btn.config(state="normal")
        if ok:
            self._set_status(f"✅  {msg}", FG_SUCCESS)
        else:
            self._set_status(f"❌  {msg}", FG_ERROR)

    def _save_and_continue(self) -> None:
        url = self._url_var.get().strip()
        token = self._token_var.get().strip()
        user = self._user_var.get().strip() or "user"

        if not url:
            self._set_status("⚠️  Gateway URL không được để trống", FG_ERROR)
            return
        if not token:
            self._set_status("⚠️  Token không được để trống", FG_ERROR)
            return

        self.config.update(
            gateway_url=url,
            gateway_token=token,
            user=user,
            setup_complete=True,
        )
        logger.info(f"[SetupWizard] Đã lưu: url={url}, user={user}")
        self._set_status("✅  Đã lưu cấu hình!", FG_SUCCESS)

        if self._root:
            self._root.after(500, self._finish)

    def _finish(self) -> None:
        if self._root:
            self._root.destroy()
            self._root = None
        self.on_complete()

    def _on_close(self) -> None:
        if self.on_cancel:
            self.on_cancel()
        if self._root:
            self._root.destroy()
            self._root = None

    def _set_status(self, text: str, color: str = FG_MUTED) -> None:
        self._status_var.set(text)
        self._status_label.config(fg=color)
