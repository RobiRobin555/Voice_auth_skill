# Thư mục chứa file .wav của người lạ (cohort) cho AS-Norm
#
# ─── Mục đích ────────────────────────────────────────────────
# AS-Norm (Adaptive Score Normalization) chuẩn hóa điểm xác thực
# bằng cách so sánh với phân phối của người lạ.
# Càng nhiều file → AS-Norm càng ổn định.
#
# ─── Yêu cầu ─────────────────────────────────────────────────
# • Định dạng : .wav (mono hoặc stereo, sẽ tự convert sang mono 16 kHz)
# • Độ dài    : 3–10 giây mỗi file
# • Số lượng  : 50–200 file (tối thiểu 20 để có ý nghĩa thống kê)
# • Nội dung  : giọng bất kỳ KHÔNG PHẢI giọng người dùng đã enroll
#
# ─── Đặt file vào đây ────────────────────────────────────────
# data/cohort/
# ├── stranger_001.wav
# ├── stranger_002.wav
# ├── stranger_003.wav
# └── ...
#
# ─── Nguồn dataset miễn phí (khuyến nghị) ───────────────────
#
# 1. LibriSpeech test-clean (tiếng Anh, ~346 người):
#    https://www.openslr.org/12/
#    wget https://www.openslr.org/resources/12/test-clean.tar.gz
#
# 2. VCTK Corpus (tiếng Anh, 110 người):
#    https://datashare.ed.ac.uk/handle/10283/3443
#
# 3. Common Voice (đa ngôn ngữ, có tiếng Việt):
#    https://commonvoice.mozilla.org/en/datasets
#    → Chọn "Vietnamese" → tải validated.tsv + clips/
#
# ─── Script cắt nhanh từ LibriSpeech ────────────────────────
# (chạy trong Python sau khi giải nén test-clean.tar.gz)
#
# import shutil, pathlib, random
# src = pathlib.Path("LibriSpeech/test-clean")
# dst = pathlib.Path("data/cohort")
# dst.mkdir(exist_ok=True)
# files = list(src.rglob("*.flac"))
# for i, f in enumerate(random.sample(files, min(100, len(files)))):
#     shutil.copy(f, dst / f"stranger_{i:03d}.flac")
# # Verifier tự convert .flac → nếu muốn .wav:
# # ffmpeg -i stranger_000.flac stranger_000.wav
#
# ─── Nếu không có cohort ─────────────────────────────────────
# Daemon vẫn chạy nhưng dùng cosine thuần thay AS-Norm.
# Threshold lúc này là cosine similarity [-1, 1], không phải Z-score.
# Khuyến nghị đặt threshold_asnorm: 0.75 khi không có cohort.
