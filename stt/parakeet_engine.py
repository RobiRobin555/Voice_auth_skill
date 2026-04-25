"""
stt/parakeet_engine.py — Wrapper Parakeet-TDT-0.6B-v3 via NVIDIA NeMo.

Tính năng:
  - Lazy-load: model chỉ tải lần đầu khi gọi transcribe().
  - Cache model trong bộ nhớ — không reload giữa các vòng lặp.
  - Hỗ trợ CPU và CUDA.
  - Ghi file WAV tạm → xóa sau khi dùng (NeMo yêu cầu file path).
  - Trả về chuỗi rỗng nếu audio quá ngắn hoặc không nhận được kết quả.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Model ID trên Hugging Face / NGC
_PARAKEET_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"


class ParakeetSTT:
    """
    Speech-to-Text sử dụng NVIDIA Parakeet-TDT-0.6B-v3.

    Parameters
    ----------
    model_id : str
        Model identifier cho NeMo (mặc định: nvidia/parakeet-tdt-0.6b-v3).
    device : str | None
        "cuda" / "cpu" / None (tự phát hiện).
    """

    def __init__(
        self,
        model_id: str = _PARAKEET_MODEL_ID,
        device: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device  # None → tự phát hiện lúc load
        self._model = None   # Lazy-load

    # ------------------------------------------------------------------
    # Lazy loader
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Tải model NeMo lần đầu (lazy)."""
        if self._model is not None:
            return

        logger.info(f"[ParakeetSTT] Đang tải model '{self.model_id}' (lần đầu ~1-2 phút) ...")

        try:
            import nemo.collections.asr as nemo_asr  # type: ignore
        except ImportError:
            raise ImportError(
                "Thiếu NeMo. Cài đặt:\n"
                "  pip install nemo_toolkit[asr]\n"
                "  (yêu cầu torch đã được cài trước)"
            )

        import torch  # type: ignore

        model = nemo_asr.models.ASRModel.from_pretrained(self.model_id)
        model.eval()

        # Chuyển sang device phù hợp
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cpu":
            model = model.cpu()
        else:
            model = model.cuda()

        self._model = model
        logger.info(f"[ParakeetSTT] Model sẵn sàng trên device='{self.device}'.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Chuyển audio numpy → văn bản.

        Parameters
        ----------
        audio : np.ndarray
            Audio float32 mono, sample_rate 16 kHz.
        sample_rate : int
            Sample rate của audio đầu vào.

        Returns
        -------
        str — văn bản đã transcribe, rỗng nếu không nhận được kết quả.
        """
        # Lazy-load model
        self._load_model()

        # Kiểm tra audio tối thiểu (0.5 giây)
        if len(audio) < sample_rate * 0.5:
            logger.debug("[ParakeetSTT] Audio quá ngắn — bỏ qua (<0.5s).")
            return ""

        # Resample nếu cần (NeMo Parakeet yêu cầu 16 kHz)
        if sample_rate != 16000:
            audio = self._resample(audio, sample_rate, 16000)

        # Normalize nhẹ để tránh clipping
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.95

        # Ghi WAV tạm → transcribe → xóa
        tmp_path = self._write_temp_wav(audio, sample_rate=16000)
        try:
            result = self._model.transcribe([tmp_path])  # type: ignore
            # NeMo trả về list; lấy phần tử đầu tiên
            if result and isinstance(result[0], str):
                text = result[0].strip()
            elif result:
                # Một số version NeMo trả về Hypothesis object
                text = str(result[0]).strip()
            else:
                text = ""
        except Exception as exc:
            logger.error(f"[ParakeetSTT] Lỗi transcribe: {exc}")
            text = ""
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if text:
            logger.debug(f"[ParakeetSTT] Kết quả: '{text}'")
        return text

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_temp_wav(audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Ghi numpy array thành file WAV tạm, trả về đường dẫn."""
        try:
            import soundfile as sf  # type: ignore
        except ImportError:
            raise ImportError("Cài soundfile:  pip install soundfile")

        fd, path = tempfile.mkstemp(suffix=".wav", prefix="parakeet_")
        os.close(fd)
        sf.write(path, audio.astype(np.float32), sample_rate, subtype="PCM_16")
        return path

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample bằng torchaudio (nếu có) hoặc scipy."""
        try:
            import torch  # type: ignore
            import torchaudio  # type: ignore

            waveform = torch.from_numpy(audio).float().unsqueeze(0)
            waveform = torchaudio.functional.resample(waveform, orig_sr, target_sr)
            return waveform.squeeze(0).numpy()
        except ImportError:
            pass

        try:
            from scipy.signal import resample_poly  # type: ignore
            from math import gcd

            g = gcd(orig_sr, target_sr)
            return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)
        except ImportError:
            logger.warning("[ParakeetSTT] Không thể resample — thiếu torchaudio và scipy.")
            return audio
