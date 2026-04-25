"""
app/openclaw_client.py — HTTP client gửi prompt đến OpenClaw Gateway.

Sử dụng REST API:
  - GET  /api/status               → kiểm tra kết nối
  - POST /api/sessions/main/messages → gửi lệnh
  - Header: Authorization: Bearer <token>
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)


class OpenClawClient:
    """
    HTTP client kết nối OpenClaw Gateway.

    Parameters
    ----------
    gateway_url : str
        URL của Gateway (vd: http://localhost:18789).
    token : str
        Bearer token để xác thực.
    timeout : int
        Timeout giây cho mỗi request.
    """

    def __init__(
        self,
        gateway_url: str = "http://localhost:18789",
        token: str = "",
        timeout: int = 30,
    ) -> None:
        self.gateway_url = gateway_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    # ── Headers ──────────────────────────────────────────────────────────

    def _headers(self, content_type: str = "application/json") -> dict[str, str]:
        h = {"Content-Type": content_type}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    # ── Test Connection ──────────────────────────────────────────────────

    def test_connection(self) -> tuple[bool, str]:
        """
        Kiểm tra kết nối đến Gateway.

        Returns
        -------
        (success, message)
        """
        url = f"{self.gateway_url}/api/status"
        logger.info(f"[OpenClawClient] Testing connection → {url}")

        req = urllib.request.Request(url, headers=self._headers(), method="GET")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                logger.info(f"[OpenClawClient] Status: {resp.status} — {body[:200]}")
                return True, f"Kết nối thành công! (HTTP {resp.status})"
        except urllib.error.HTTPError as exc:
            msg = f"HTTP {exc.code}: {exc.reason}"
            logger.warning(f"[OpenClawClient] {msg}")
            return False, msg
        except urllib.error.URLError as exc:
            msg = f"Không kết nối được: {exc.reason}"
            logger.warning(f"[OpenClawClient] {msg}")
            return False, msg
        except Exception as exc:
            msg = f"Lỗi: {exc}"
            logger.error(f"[OpenClawClient] {msg}")
            return False, msg

    # ── Send Prompt ──────────────────────────────────────────────────────

    def send_prompt(self, text: str, session: str = "main") -> tuple[bool, str]:
        """
        Gửi prompt text đến OpenClaw Gateway.

        Parameters
        ----------
        text : str
            Lệnh/câu hỏi đã transcribe.
        session : str
            Session ID (mặc định: "main").

        Returns
        -------
        (success, response_text)
        """
        if not text.strip():
            return False, "Prompt rỗng"

        url = f"{self.gateway_url}/api/sessions/{session}/messages"
        payload = json.dumps({
            "message": text,
            "source": "voice_auth_tray",
        }).encode("utf-8")

        logger.info(f"[OpenClawClient] POST {url} — prompt: '{text[:80]}'")

        req = urllib.request.Request(
            url,
            data=payload,
            headers=self._headers(),
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                logger.info(f"[OpenClawClient] Response {resp.status}: {body[:200]}")

                try:
                    data = json.loads(body)
                    result = (
                        data.get("result")
                        or data.get("response")
                        or data.get("message")
                        or body
                    )
                except json.JSONDecodeError:
                    result = body

                return True, str(result)

        except urllib.error.HTTPError as exc:
            msg = f"HTTP {exc.code}: {exc.reason}"
            try:
                err_body = exc.read().decode("utf-8")[:300]
                msg += f" — {err_body}"
            except Exception:
                pass
            logger.error(f"[OpenClawClient] {msg}")
            return False, msg

        except urllib.error.URLError as exc:
            msg = f"Không kết nối được Gateway: {exc.reason}"
            logger.error(f"[OpenClawClient] {msg}")
            return False, msg

        except Exception as exc:
            msg = f"Lỗi gửi prompt: {exc}"
            logger.error(f"[OpenClawClient] {msg}")
            return False, msg

    # ── Update credentials ───────────────────────────────────────────────

    def update(self, gateway_url: str, token: str) -> None:
        """Cập nhật URL và token runtime (khi user sửa settings)."""
        self.gateway_url = gateway_url.rstrip("/")
        self.token = token
