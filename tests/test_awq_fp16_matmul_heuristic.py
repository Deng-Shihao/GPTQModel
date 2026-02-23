from types import SimpleNamespace

import pytest
import torch

import gptqmodel.nn_modules.qlinear.gemm_awq as gemm_awq
import gptqmodel.nn_modules.qlinear.gemm_awq_triton as gemm_awq_triton


def _fake_quant_tensors(in_features: int = 32, out_features: int = 8, group_size: int = 32):
    qweight = torch.ones((in_features, out_features // 8), dtype=torch.int32)
    scales = torch.ones((in_features // group_size, out_features), dtype=torch.float16)
    qzeros = torch.zeros((in_features // group_size, out_features // 8), dtype=torch.int32)
    return qweight, scales, qzeros


def _patch_backend(monkeypatch, backend: str, calls):
    if backend == "triton":
        monkeypatch.setattr(gemm_awq, "awq_ext", None)

        triton_state = getattr(gemm_awq_triton, "tritonv2", SimpleNamespace(TRITON_AVAILABLE=False))
        monkeypatch.setattr(gemm_awq_triton, "tritonv2", triton_state, raising=False)
        monkeypatch.setattr(triton_state, "TRITON_AVAILABLE", True)

        def fake_dequant(qweight, scales, qzeros):
            calls["dequant"] += 1
            return torch.ones(qweight.shape[0], qweight.shape[1] * 8, dtype=torch.float16)

        def fake_gemm(input, qweight, scales, qzeros, split_k_iters, **_):
            calls["gemm"] += 1
            out_features = qweight.shape[1] * 8
            return torch.ones(input.shape[0], out_features, device=input.device, dtype=input.dtype)

        monkeypatch.setattr(gemm_awq_triton, "awq_dequantize_triton", fake_dequant, raising=False)
        monkeypatch.setattr(gemm_awq_triton, "awq_gemm_triton", fake_gemm, raising=False)
        monkeypatch.setattr(
            "gptqmodel.quantization.awq.modules.triton.gemm.awq_dequantize_triton",
            fake_dequant,
            raising=False,
        )
        monkeypatch.setattr(
            "gptqmodel.quantization.awq.modules.triton.gemm.awq_gemm_triton",
            fake_gemm,
            raising=False,
        )

        return gemm_awq_triton.AwqGemmTritonFn

    # Stub the compiled AWQ extension so we can count which path is taken.
    class FakeAwqExt:
        def dequantize_weights_cuda(self, qweight, scales, qzeros, *_args):
            calls["dequant"] += 1
            return torch.ones(qweight.shape[0], qweight.shape[1] * 8, dtype=torch.float16)

        def gemm_forward_cuda(self, input, qweight, scales, qzeros, _split_k_iters):
            calls["gemm"] += 1
            out_features = qweight.shape[1] * 8
            return torch.ones(input.shape[0], out_features, device=input.device, dtype=input.dtype)

    monkeypatch.setattr(gemm_awq, "awq_ext", FakeAwqExt())
    triton_state = getattr(gemm_awq_triton, "tritonv2", SimpleNamespace(TRITON_AVAILABLE=False))
    monkeypatch.setattr(gemm_awq_triton, "tritonv2", triton_state, raising=False)
    monkeypatch.setattr(triton_state, "TRITON_AVAILABLE", False)
    return gemm_awq.AwqGemmFn


@pytest.mark.parametrize("backend", ["triton", "ext"], ids=["triton", "awq_ext"])
def test_fp16_matmul_heuristic_prefers_dequant_for_large_matrices(monkeypatch, backend):
    calls = {"dequant": 0, "gemm": 0}
    fn = _patch_backend(monkeypatch, backend, calls)

    group_size = 32
    out_features = 8
    qweight, scales, qzeros = _fake_quant_tensors(in_features=32, out_features=out_features, group_size=group_size)

    # Large batch x sequence (33*32=1056 rows) exceeds the 1024-row heuristic
    # and activates the dequantize-then-matmul path.
    x = torch.ones((33, 32, qweight.shape[0]), dtype=torch.float16)

    out = fn.apply(
        x, qweight, qzeros, scales, 4, group_size, None, out_features,
    )

    assert calls == {"dequant": 1, "gemm": 0}
    assert out.shape == (33, 32, out_features)


@pytest.mark.parametrize("backend", ["triton", "ext"], ids=["triton", "awq_ext"])
def test_fp16_matmul_heuristic_prefers_fused_gemm_for_small_matrices(monkeypatch, backend):
    calls = {"dequant": 0, "gemm": 0}
    fn = _patch_backend(monkeypatch, backend, calls)

    group_size = 32
    out_features = 8
    qweight, scales, qzeros = _fake_quant_tensors(in_features=32, out_features=out_features, group_size=group_size)

    # Small batch x sequence (1 row) sits below the 1024-row heuristic and
    # stays on the fused GEMM kernel.
    x = torch.ones((1, 1, qweight.shape[0]), dtype=torch.float16)

    out = fn.apply(
        x, qweight, qzeros, scales, 4, group_size, None, out_features,
    )

    assert calls == {"dequant": 0, "gemm": 1}
    assert out.shape == (1, 1, out_features)
