# src/models — Thư mục chứa kiến trúc model
#
# Đặt file ecapa_tdnn.py của bạn vào đây:
#   src/models/ecapa_tdnn.py
#
# File này là source code định nghĩa class ECAPA_TDNN (C=512, emb_size=192).
# Bạn có thể copy thẳng từ project train của mình.
#
# ─── Giao diện tối thiểu cần có ──────────────────────────────
#
#   class ECAPA_TDNN(nn.Module):
#       def __init__(self, C: int = 512, ...):
#           ...
#
#       def forward(self, x: Tensor) -> Tensor:
#           # x: (batch, time) hoặc (batch, n_mels, time)
#           # return: (batch, emb_size=192)
#           ...
#
# ─── Nếu không có ecapa_tdnn.py ──────────────────────────────
# Verifier sẽ fallback sang torch.load(model_path) — load model
# đã serialize hoàn chỉnh (torch.save(model, path)).
# Trong trường hợp này không cần file này.
