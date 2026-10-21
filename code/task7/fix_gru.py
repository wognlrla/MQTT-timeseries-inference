"""과제 7: GRU 예측 모델의 FP16 오차(최대 1.31ppm > 기준 1ppm)를 줄이는 정밀도 조합을 찾는다.

변형
- fp16          : 전부 FP16 허용 (기본)
- fp32_noTF32   : FP32, TF32도 끔 (가장 정확한 기준)
- fp16_headFP32 : GRU는 FP16, 출력 MLP(head)는 FP32 고정
- fp16_gruFP32  : GRU는 FP32 고정, head는 FP16
각 변형의 CO₂ 최대·평균 차이(ppm, PyTorch CPU 대비)와 지연(배치 1·4)을 잰다.
"""
import sys
import time

import numpy as np
import pandas as pd
import tensorrt as trt
import torch

sys.path.insert(0, "/srv/team/task7")
import bench as B  # noqa: E402

torch.set_grad_enabled(False)


def build(onnx_path, spec, mode):
    builder = trt.Builder(B.LOG)
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, B.LOG)
    assert parser.parse(onnx_path.read_bytes())
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    if mode.startswith("fp16"):
        cfg.set_flag(trt.BuilderFlag.FP16)
    if mode == "fp32_noTF32":
        cfg.clear_flag(trt.BuilderFlag.TF32)
    keep = {"fp16_headFP32": "head", "fp16_gruFP32": "gru"}.get(mode)
    n_fixed = 0
    if keep:
        cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        for i in range(net.num_layers):
            L = net.get_layer(i)
            floats = all(L.get_output(j).dtype in (trt.float32, trt.float16) for j in range(L.num_outputs))
            if keep in L.name.lower() and floats and L.type not in (trt.LayerType.CONSTANT, trt.LayerType.SHAPE):
                L.precision = trt.float32
                for j in range(L.num_outputs):
                    if L.get_output(j).dtype in (trt.float32, trt.float16):
                        L.set_output_type(j, trt.float32)
                n_fixed += 1
    prof = builder.create_optimization_profile()
    for n, a in zip(spec["names"], spec["inputs"]):
        feat = tuple(a.shape[1:])
        prof.set_shape(n, (1,) + feat, (4,) + feat, (B.MAX_B,) + feat)
    cfg.add_optimization_profile(prof)
    ser = builder.build_serialized_network(net, cfg)
    path = B.ENG / f"forecast_gru.{mode}.engine"
    path.write_bytes(bytes(ser))
    names = [net.get_layer(i).name for i in range(min(net.num_layers, 400))]
    return path, n_fixed, net.num_layers, names


def main():
    spec = B.prepare()["forecast_gru"]
    onnx_path = B.ENG / "forecast_gru.onnx"
    ref = B.full_output(lambda *a: spec["model"].cpu()(*[torch.from_numpy(x) for x in a]).numpy(), spec["inputs"])
    std = spec["extra"]["std"]
    rows = []
    modes = sys.argv[1:] or ["fp16", "fp32_noTF32", "fp16_headFP32", "fp16_gruFP32"]
    for mode in modes:
        path, n_fixed, n_layers, names = build(onnx_path, spec, mode)
        if mode == "fp16":
            print("레이어 이름 예:", [n for n in names if "gru" in n.lower()][:3], [n for n in names if "head" in n.lower()][:3])
        r = B.TRTRunner(path)
        out = B.full_output(r, spec["inputs"])
        d = np.abs(out - ref) * std
        lat = {}
        for b in (1, 4):
            args = [np.ascontiguousarray(a[:b]) for a in spec["inputs"]]
            m, p95, _ = B.bench(r, args)
            lat[f"b{b} 평균/p95(ms)"] = f"{m} / {p95}"
        rows.append({"변형": mode, "FP32 고정 레이어": f"{n_fixed}/{n_layers}",
                     "CO2 최대 차이(ppm)": round(float(d[:, :, 0].max()), 4), "CO2 평균 차이(ppm)": round(float(d[:, :, 0].mean()), 5),
                     "CO2 99.9% 차이(ppm)": round(float(np.quantile(d[:, :, 0], 0.999)), 4),
                     "온도 최대 차이(℃)": round(float(d[:, :, 1].max()), 5), "엔진(KB)": round(path.stat().st_size / 1024, 1)} | lat)
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    out = B.OUT / "gru_precision.csv"
    if out.exists() and sys.argv[1:]:
        df = pd.concat([pd.read_csv(out).query("변형 not in @modes"), df], ignore_index=True)
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
