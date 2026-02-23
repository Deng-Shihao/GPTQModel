import pytest

from awequant.models.auto import AweQuant
from awequant.quantization.config import FORMAT, METHOD, QuantizeConfig


def test_quantize_config_defaults_to_awq():
    cfg = QuantizeConfig()
    assert cfg.quant_method == METHOD.AWQ
    assert cfg.format == FORMAT.GEMM


def test_quantize_config_rejects_non_awq_methods():
    with pytest.raises(ValueError, match="AWQ-only"):
        QuantizeConfig(quant_method="gptq", format="gptq")


def test_from_quant_config_rejects_non_awq():
    payload = {
        "bits": 4,
        "group_size": 128,
        "quant_method": "gptq",
        "checkpoint_format": "gptq",
    }
    with pytest.raises(ValueError, match="AWQ-only"):
        QuantizeConfig.from_quant_config(payload)


def test_awq_only_disables_eval_export_and_hub_helpers():
    with pytest.raises(NotImplementedError, match="AWQ-only"):
        AweQuant.eval()

    with pytest.raises(NotImplementedError, match="AWQ-only"):
        AweQuant.export("model", "out", "hf")

    with pytest.raises(NotImplementedError, match="AWQ-only"):
        AweQuant.push_to_hub("repo", "path")

