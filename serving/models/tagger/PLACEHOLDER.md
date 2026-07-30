# Placeholder model

`model.onnx` is currently the **NERP benchmark model** (B1, indobert-lite-p2),
trained on Indonesian news NER, not on order messages.

It is here so the serving layer runs end to end before the real student exists.
Its predictions are meaningless for order parsing.

Replace with the distilled student from `training/order/` when available,
then delete this file.

See docs/RESULTS.md