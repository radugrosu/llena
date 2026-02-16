import subprocess
import sys

import pytest


def test_liger_replaces_qwen2_input_layernorm_class() -> None:
    pytest.importorskip("liger_kernel")

    # Run in a subprocess because applying Liger mutates transformer classes globally.
    script = r"""
from transformers import Qwen2Config, Qwen2ForCausalLM
from mm.model import _apply_liger_kernel_to_qwen2

def tiny_qwen2() -> Qwen2ForCausalLM:
    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    return Qwen2ForCausalLM(cfg)

base_model = tiny_qwen2()
base_cls = base_model.model.layers[0].input_layernorm.__class__.__name__
_apply_liger_kernel_to_qwen2()
liger_model = tiny_qwen2()
liger_cls = liger_model.model.layers[0].input_layernorm.__class__.__name__
assert "Liger" not in base_cls, base_cls
assert "Liger" in liger_cls, liger_cls
"""
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
