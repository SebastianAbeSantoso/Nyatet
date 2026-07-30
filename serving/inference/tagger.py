import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from .spans import collapse_bio


class SpanTagger:
    def __init__(self, model_dir: str | Path, num_threads: int = 1):
        model_dir = Path(model_dir)

        onnx_files = sorted(model_dir.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(
                f"No .onnx file in {model_dir}. Export a model there first "
                f"(see training/ notebooks), or check the MODEL_DIR env var."
            )

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads  
        self.session = ort.InferenceSession(str(onnx_files[0]), opts)
        self.input_names = {i.name for i in self.session.get_inputs()}

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        config = json.loads((model_dir / "config.json").read_text())
        id2label = config.get("id2label")
        if not id2label:
            raise ValueError(
                f"config.json in {model_dir} has no id2label. The export step "
                f"must pass id2label/label2id to from_pretrained."
            )
        self.id2label = {int(k): v for k, v in id2label.items()}

    @property
    def labels(self) -> list[str]:
        return [self.id2label[i] for i in sorted(self.id2label)]

    def tag(self, text: str, max_length: int = 128) -> list[dict]:
        enc = self.tokenizer(
            text, return_tensors="np", truncation=True, max_length=max_length
        )
        feed = {n: enc[n].astype(np.int64) for n in self.input_names if n in enc}
        missing = self.input_names - set(feed)
        if missing:
            raise RuntimeError(f"Model expects inputs the tokenizer did not produce: {missing}")

        logits = self.session.run(None, feed)[0][0]      
        probs = _softmax(logits)
        tag_ids = probs.argmax(axis=-1)
        confidences = probs.max(axis=-1)

        tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        special = set(self.tokenizer.all_special_tokens)

        keep = [i for i, t in enumerate(tokens) if t not in special]
        return collapse_bio(
            [tokens[i] for i in keep],
            [self.id2label[int(tag_ids[i])] for i in keep],
            [float(confidences[i]) for i in keep],
        )


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)