import pytest
import importlib
from transformers import Qwen2Config, Qwen2ForCausalLM
import transformers.models.qwen2.modeling_qwen2 as qwen_modeling


# This fixture is the "Cleanup Crew"
@pytest.fixture(autouse=True)
def clean_qwen2_state():
    # 1. Yield to let the test run
    yield
    # 2. Discard the Liger monkey-patches and restores standard PyTorch classes
    importlib.reload(qwen_modeling)


def test_liger_replaces_qwen2_input_layernorm_class():
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2

    # Helper to build a tiny model
    def build_tiny():
        cfg = Qwen2Config(
            vocab_size=100,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
        return Qwen2ForCausalLM(cfg)

    # 1. Verify Baseline (Standard PyTorch)
    base_model = build_tiny()
    # Check strict class equality, not just string name
    assert "Qwen2RMSNorm" in base_model.model.layers[0].input_layernorm.__class__.__name__

    # 2. Apply the Patch
    apply_liger_kernel_to_qwen2()

    # 3. Verify Liger (Patched)
    liger_model = build_tiny()
    layer_cls_name = liger_model.model.layers[0].input_layernorm.__class__.__name__

    # Depending on Liger version, it might be LigerRMSNorm or similar
    assert "Liger" in layer_cls_name, f"Expected Liger layer, got {layer_cls_name}"
