# Thư mục chứa model ECAPA-TDNN đã train
#
# Đặt file model vào đây rồi cập nhật config.yaml:
#
#   models/
#   └── ecapa_tdnn.pt        ← file checkpoint của bạn (.pt hoặc .pth)
#
# ─── Định dạng model được hỗ trợ ─────────────────────────────
#
#   Dạng 1 — State dict (phổ biến khi train bằng PyTorch):
#     checkpoint = {
#         "model_state_dict": OrderedDict({ ... }),
#         "epoch": 100,
#         ...
#     }
#     torch.save(checkpoint, "ecapa_tdnn.pt")
#
#   Dạng 2 — Module serialized trực tiếp:
#     torch.save(model, "ecapa_tdnn.pt")
#
# ─── Kiến trúc mong đợi ──────────────────────────────────────
#
#   ECAPA_TDNN(C=512)  →  embedding 192-dim
#   File nguồn:  src/models/ecapa_tdnn.py  (từ project của bạn)
#
#   Nếu không có src/models/ecapa_tdnn.py → verifier tự fallback
#   sang torch.load() (dạng module serialized).
#
# ─── Cập nhật config.yaml sau khi đặt file ───────────────────
#
#   auth:
#     model_path: models/ecapa_tdnn.pt   ← tên file của bạn
#
# ─── Kiểm tra nhanh ──────────────────────────────────────────
#
#   python -c "
#   import sys; sys.path.insert(0, '.')
#   from auth.verifier import VoiceVerifier
#   print('Model path OK')
#   "
