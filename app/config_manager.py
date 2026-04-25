"""
app/config_manager.py — Quản lý cấu hình persistent cho Voice Auth Tray.

Lưu settings tại:
  Windows: %APPDATA%/VoiceAuthTray/settings.json
  Linux:   ~/.config/VoiceAuthTray/settings.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Thư mục gốc của ứng dụng (nơi chứa models/, data/, auth/, ...) ────────
APP_ROOT = Path(__file__).resolve().parent.parent


def _appdata_dir() -> Path:
    """Trả về thư mục lưu settings tuỳ hệ điều hành."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "VoiceAuthTray"


# ── Default settings ──────────────────────────────────────────────────────

_DEFAULTS: dict[str, Any] = {
    # OpenClaw Gateway
    "gateway_url": "http://localhost:18789",
    "gateway_token": "",

    # User
    "user": "user",

    # Auth
    "model_path": str(APP_ROOT / "models" / "model.pt"),
    "voiceprint_path": "",          # sẽ được set khi enroll
    "cohort_dir": str(APP_ROOT / "data" / "cohort"),
    "threshold_asnorm": 1.5,
    "max_auth_duration_sec": 4,

    # Wake word
    "wake_word_enabled": False,
    "wake_word_engine": "openwakeword",
    "wake_word_keyword": "hey openclaw",

    # Logging
    "log_path": str(APP_ROOT / "logs" / "auth_log.jsonl"),

    # State flags
    "setup_complete": False,
    "enrolled": False,
}


class ConfigManager:
    """
    Load / save persistent settings.

    Settings lưu tại %APPDATA%/VoiceAuthTray/settings.json.
    Mọi đường dẫn tương đối tự resolve từ APP_ROOT.
    """

    def __init__(self, settings_path: Optional[str] = None) -> None:
        if settings_path:
            self._path = Path(settings_path)
        else:
            self._path = _appdata_dir() / "settings.json"

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = dict(_DEFAULTS)
        self._load()

    # ── Load / Save ──────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            logger.info(f"[Config] No settings found, creating at '{self._path}'")
            self._save()
            return
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
            # Merge: giữ defaults cho key mới, ghi đè bằng saved
            self._data.update(saved)
            logger.info(f"[Config] Loaded settings from '{self._path}'")
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"[Config] Error reading settings: {exc} - using defaults")

    def _save(self) -> None:
        try:
            with open(self._path, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2, ensure_ascii=False)
        except OSError as exc:
            logger.error(f"[Config] Cannot write settings: {exc}")

    def save(self) -> None:
        """Public save — ghi settings ra file."""
        self._save()

    # ── Getters / Setters ────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def update(self, **kwargs: Any) -> None:
        """Cập nhật nhiều key cùng lúc rồi save."""
        self._data.update(kwargs)
        self._save()

    # ── Convenience properties ───────────────────────────────────────────

    @property
    def gateway_url(self) -> str:
        return self._data.get("gateway_url", _DEFAULTS["gateway_url"])

    @property
    def gateway_token(self) -> str:
        return self._data.get("gateway_token", "")

    @property
    def user(self) -> str:
        return self._data.get("user", "user")

    @property
    def setup_complete(self) -> bool:
        return bool(self._data.get("setup_complete", False))

    @property
    def enrolled(self) -> bool:
        return bool(self._data.get("enrolled", False))

    @property
    def model_path(self) -> str:
        p = self._data.get("model_path", "")
        return self._resolve(p) if p else str(APP_ROOT / "models" / "model.pt")

    @property
    def voiceprint_path(self) -> str:
        p = self._data.get("voiceprint_path", "")
        if p:
            return self._resolve(p)
        return str(APP_ROOT / "auth" / "voiceprints" / f"{self.user}.npy")

    @property
    def cohort_dir(self) -> str:
        p = self._data.get("cohort_dir", "")
        return self._resolve(p) if p else str(APP_ROOT / "data" / "cohort")

    @property
    def threshold(self) -> float:
        return float(self._data.get("threshold_asnorm", 1.5))

    @property
    def log_path(self) -> str:
        p = self._data.get("log_path", "")
        return self._resolve(p) if p else str(APP_ROOT / "logs" / "auth_log.jsonl")

    # ── Path resolution ──────────────────────────────────────────────────

    @staticmethod
    def _resolve(path: str) -> str:
        """Nếu path tương đối → resolve từ APP_ROOT."""
        p = Path(path)
        if p.is_absolute():
            return str(p)
        return str(APP_ROOT / p)

    # ── Reset ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset về defaults."""
        self._data = dict(_DEFAULTS)
        self._save()
        logger.info("[Config] Reset to defaults.")

    def __repr__(self) -> str:
        return f"ConfigManager(path='{self._path}', setup={self.setup_complete}, enrolled={self.enrolled})"
