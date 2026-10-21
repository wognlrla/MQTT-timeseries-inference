"""과제 7: 학습 모델을 ONNX → TensorRT(FP16)로 변환하고, 실행 방식별 지연·결과 차이·전력·메모리를 잰다.

모델: 예측 GRU(과제 3) / 이상 탐지 오토인코더(과제 4) / 사용 여부·환기 경보 MLP(과제 5)
      + TCN(학습 안 한 같은 크기 모델, 지연 비교용 — 'GRU vs TCN' 질문)
실행 방식: PyTorch CPU / PyTorch GPU / ONNX Runtime(CPU) / TensorRT FP32(참고) / TensorRT FP16
지연: 입력(numpy, 호스트) → 결과(numpy, 호스트)까지 전체. 워밍업 30회 후 블록당 3초 동안 반복, 평균·p95.
결과 차이: 시험 기간 실제 입력 전체, PyTorch CPU(FP32) 출력 대비, 원래 단위(ppm·℃·점수·분류 결과).
전력·GPU·메모리: 블록마다 tegrastats(100ms 간격) 평균.

사용: /srv/team/venv/bin/python bench.py
"""
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, "/srv/team/common")
sys.path.insert(0, "/srv/team/task3")
sys.path.insert(0, "/srv/team/task4")
import preprocess as pp  # noqa: E402
import features as F  # noqa: E402
import detectors as D  # noqa: E402
from train import GRUForecaster  # noqa: E402
import onnxruntime as ort  # noqa: E402
import tensorrt as trt  # noqa: E402

ROOT = Path("/srv/team/task7")
ENG = ROOT / "engines"
OUT = ROOT / "out"
LOG = trt.Logger(trt.Logger.WARNING)
MAX_B = 4096
SECS = 3.0
torch.set_grad_enabled(False)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ 모델과 실제 입력
class TCNForecaster(nn.Module):
    """GRU와 같은 입출력의 인과 합성곱 모델(지연 비교용, 학습 안 함)."""
    def __init__(self, f_past, f_static, f_future, ch=48, horizon=F.HORIZON, n_out=2):
        super().__init__()
        self.horizon, self.n_out = horizon, n_out
        layers, c_in = [], f_past
        for d in (1, 2, 4, 8):
            layers += [nn.ConstantPad1d((2 * d, 0), 0.0), nn.Conv1d(c_in, ch, 3, dilation=d), nn.ReLU()]
            c_in = ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(ch + f_static + f_future * horizon, 128), nn.ReLU(),
                                  nn.Linear(128, horizon * n_out))

    def forward(self, past, static, future):
        h = self.tcn(past.transpose(1, 2))[:, :, -1]
        return self.head(torch.cat([h, static, future.flatten(1)], 1)).view(-1, self.horizon, self.n_out)


def prepare():
    grid, raw, _, schedule, faults, meta = pp.build()
    norm = json.loads(Path("/srv/team/task2/out/norm_stats.json").read_text())["stats"]
    models = {}

    ck = torch.load("/srv/team/task3/models/gru_sched_s0.pt", weights_only=False)
    gru = GRUForecaster(*ck["config"]["dims"]).eval()
    gru.load_state_dict(ck["state_dict"])
    arr = F.build_arrays(grid, schedule, meta, norm, use_schedule=True)
    sel = np.where((arr["split"] == "test") & arr["cur_ok"])[0]
    fx = [np.ascontiguousarray(arr[k][sel]) for k in ("past", "static", "future")]
    models["forecast_gru"] = dict(model=gru, inputs=fx, names=["past", "static", "future"],
                                  extra={"cur": arr["cur"][sel], "std": arr["std_t"]})
    torch.manual_seed(0)
    models["forecast_tcn(지연만)"] = dict(model=TCNForecaster(*ck["config"]["dims"]).eval(), inputs=fx,
                                         names=["past", "static", "future"], extra=None)

    ae = torch.load("/srv/team/task4/ae.pt", weights_only=False)
    m = D.CondAE(*ae["dims"]).eval()
    m.load_state_dict(ae["state_dict"])
    w = D.ae_windows(grid, schedule, meta, norm)
    s = np.where((w["split"] == "test") & w["ok"])[0]
    models["anomaly_ae"] = dict(model=m, inputs=[np.ascontiguousarray(w["x"][s]), np.ascontiguousarray(w["c"][s])],
                                names=["x", "c"], extra={"scale": np.array(ae["scale"], np.float32)})

    t5 = load_module("t5run", "/srv/team/task5/run.py")
    df, _, _ = t5.build_table()
    ok = df.co2.notna() & df.dco2_30.notna() & ~df.fault_near & (df.split == "test")
    te = df[ok]
    for key, path in [("occupancy_mlp", "/srv/team/task5/occ_mlp.pt"), ("vent_mlp", "/srv/team/task5/vent_mlp.pt")]:
        c = torch.load(path, weights_only=False)
        mean, std = pd.Series(c["norm"]["mean"]), pd.Series(c["norm"]["std"])
        X = ((te[c["cols"]] - mean[c["cols"]]) / std[c["cols"]]).fillna(0).to_numpy(np.float32)
        mm = t5.MLP(X.shape[1]).eval()
        mm.load_state_dict(c["state_dict"])
        models[key] = dict(model=mm, inputs=[np.ascontiguousarray(X)], names=["x"],
                           extra={"threshold": c.get("threshold", 0.5)})
    return models


# ------------------------------------------------------------------ 변환
def export_onnx(key, spec):
    path = ENG / f"{key.split('(')[0]}.onnx"
    ex = tuple(torch.from_numpy(a[:4]) for a in spec["inputs"])
    torch.onnx.export(spec["model"], ex, str(path), input_names=spec["names"], output_names=["out"],
                      dynamic_axes={n: {0: "batch"} for n in spec["names"] + ["out"]}, opset_version=17, dynamo=False)
    return path


def build_trt(onnx_path, spec, fp16):
    builder = trt.Builder(LOG)
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, LOG)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    if fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
    prof = builder.create_optimization_profile()
    for n, a in zip(spec["names"], spec["inputs"]):
        feat = tuple(a.shape[1:])
        prof.set_shape(n, (1,) + feat, (4,) + feat, (MAX_B,) + feat)
    cfg.add_optimization_profile(prof)
    t0 = time.time()
    ser = builder.build_serialized_network(net, cfg)
    if ser is None:
        raise RuntimeError("engine build failed")
    path = onnx_path.with_suffix(".fp16.engine" if fp16 else ".fp32.engine")
    path.write_bytes(bytes(ser))
    return path, time.time() - t0


class TRTRunner:
    def __init__(self, path):
        self.rt = trt.Runtime(LOG)
        self.eng = self.rt.deserialize_cuda_engine(path.read_bytes())
        self.ctx = self.eng.create_execution_context()
        names = [self.eng.get_tensor_name(i) for i in range(self.eng.num_io_tensors)]
        self.ins = [n for n in names if self.eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.out = [n for n in names if self.eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT][0]
        self.stream = torch.cuda.Stream()

    def __call__(self, *arrs):
        with torch.cuda.stream(self.stream):
            ts = [torch.from_numpy(a).cuda(non_blocking=True) for a in arrs]
            for n, t in zip(self.ins, ts):
                self.ctx.set_input_shape(n, tuple(t.shape))
                self.ctx.set_tensor_address(n, t.data_ptr())
            o = torch.empty(tuple(self.ctx.get_tensor_shape(self.out)), dtype=torch.float32, device="cuda")
            self.ctx.set_tensor_address(self.out, o.data_ptr())
            self.ctx.execute_async_v3(self.stream.cuda_stream)
            res = o.cpu()
        self.stream.synchronize()
        return res.numpy()

    def device_mb(self):
        return self.eng.device_memory_size_v2 / 2**20


def runners(key, spec):
    cpu = spec["model"].cpu()
    gpu = __import__("copy").deepcopy(spec["model"]).cuda()
    onnx_path = export_onnx(key, spec)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    e32, s32 = build_trt(onnx_path, spec, fp16=False)
    e16, s16 = build_trt(onnx_path, spec, fp16=True)
    r32, r16 = TRTRunner(e32), TRTRunner(e16)
    names = spec["names"]

    def t_gpu(*a):
        out = gpu(*[torch.from_numpy(x).cuda() for x in a]).cpu()
        return out.numpy()
    run = {
        "PyTorch CPU": lambda *a: cpu(*[torch.from_numpy(x) for x in a]).numpy(),
        "PyTorch GPU": t_gpu,
        "ONNX Runtime (CPU)": lambda *a: sess.run(None, dict(zip(names, a)))[0],
        "TensorRT FP32": r32,
        "TensorRT FP16": r16,
    }
    info = {"onnx_KB": round(onnx_path.stat().st_size / 1024, 1),
            "engine_fp16_KB": round(e16.stat().st_size / 1024, 1), "build_fp16_s": round(s16, 1),
            "trt_fp16_activation_MB": round(r16.device_mb(), 2), "params": sum(p.numel() for p in cpu.parameters())}
    torch.cuda.reset_peak_memory_stats()
    t_gpu(*[a[:4] for a in spec["inputs"]])
    info["torch_gpu_peak_MB"] = round(torch.cuda.max_memory_allocated() / 2**20, 2)
    return run, info


# ------------------------------------------------------------------ 측정
class Tegra:
    def __enter__(self):
        self.p = subprocess.Popen(["tegrastats", "--interval", "100"], stdout=subprocess.PIPE, text=True)
        return self

    def __exit__(self, *a):
        self.p.terminate()
        txt = self.p.communicate()[0]
        def avg(pat):
            v = [float(x) for x in re.findall(pat, txt)]
            return round(float(np.mean(v)), 1) if v else None
        self.res = {"VDD_IN_mW": avg(r"VDD_IN (\d+)mW"), "VDD_CPU_GPU_CV_mW": avg(r"VDD_CPU_GPU_CV (\d+)mW"),
                    "GPU_util_%": avg(r"GR3D_FREQ (\d+)%"), "RAM_used_MB": avg(r"RAM (\d+)/\d+MB")}


def bench(fn, args):
    for _ in range(30):
        fn(*args)
    ts, end = [], time.time() + SECS
    while time.time() < end:
        t0 = time.perf_counter()
        fn(*args)
        ts.append((time.perf_counter() - t0) * 1000)
    return round(float(np.mean(ts)), 3), round(float(np.percentile(ts, 95)), 3), len(ts)


def full_output(fn, inputs, bs=2048):
    return np.concatenate([fn(*[a[i:i + bs] for a in inputs]) for i in range(0, len(inputs[0]), bs)])


def diffs(key, spec, outs):
    ref = outs["PyTorch CPU"]
    res = {}
    for m, o in outs.items():
        if m == "PyTorch CPU":
            continue
        if key == "forecast_gru":
            d = np.abs(o - ref) * spec["extra"]["std"]  # 변화량(정규화) 차이 → ppm, ℃
            res[m] = {"CO2 최대 차이(ppm)": round(float(d[:, :, 0].max()), 4), "CO2 평균 차이(ppm)": round(float(d[:, :, 0].mean()), 5),
                      "온도 최대 차이(℃)": round(float(d[:, :, 1].max()), 5)}
        elif key == "anomaly_ae":
            x = spec["inputs"][0]
            def score(r):
                e = ((r - x) ** 2).reshape(len(x), D.WIN, 3)[:, -D.SCORE_LAST:, :].mean(1) / spec["extra"]["scale"]
                return e.max(1)
            s0, s1 = score(ref), score(o)
            res[m] = {"점수 최대 차이": round(float(np.abs(s1 - s0).max()), 5),
                      "경보 판정 바뀐 비율(%)": round(100 * float(((s0 > 1.5) != (s1 > 1.5)).mean()), 4)}
        else:
            thr = spec["extra"]["threshold"]
            p0, p1 = 1 / (1 + np.exp(-ref)), 1 / (1 + np.exp(-o))
            res[m] = {"확률 최대 차이": round(float(np.abs(p1 - p0).max()), 6),
                      "분류 바뀐 비율(%)": round(100 * float(((p0 >= thr) != (p1 >= thr)).mean()), 4)}
    return res


def main():
    ENG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cond = {"power_mode": os.popen("nvpmodel -q 2>/dev/null | head -1").read().strip(),
            "load_avg": open("/proc/loadavg").read().split()[:3], "clocks": "고정 안 함(jetson_clocks 미실행, DVFS)",
            "warmup": 30, "block_seconds": SECS, "timing": "host numpy 입력 → host numpy 결과",
            "tensorrt": trt.__version__, "torch": torch.__version__, "onnxruntime": ort.__version__}
    with Tegra() as idle:
        time.sleep(3)
    cond["idle"] = idle.res
    print(json.dumps(cond, ensure_ascii=False), flush=True)

    models = prepare()
    lat_rows, diff_rows, info_rows = [], [], []
    for key, spec in models.items():
        run, info = runners(key, spec)
        info_rows.append({"model": key} | info)
        if spec["extra"] is not None:
            outs = {m: full_output(fn, spec["inputs"]) for m, fn in run.items()}
            for m, d in diffs(key, spec, outs).items():
                diff_rows.append({"model": key, "method": m, "n": len(spec["inputs"][0])} | d)
        for b in (1, 4):
            args = [np.ascontiguousarray(a[:b]) for a in spec["inputs"]]
            for m, fn in run.items():
                with Tegra() as tg:
                    mean, p95, n = bench(fn, args)
                lat_rows.append({"model": key, "batch": b, "method": m, "mean_ms": mean, "p95_ms": p95, "iters": n} | tg.res)
                print(key, b, m, mean, p95, tg.res, flush=True)
    lat, dif, inf = pd.DataFrame(lat_rows), pd.DataFrame(diff_rows), pd.DataFrame(info_rows)
    lat.to_csv(OUT / "latency.csv", index=False)
    dif.to_csv(OUT / "diff.csv", index=False)
    inf.to_csv(OUT / "model_info.csv", index=False)
    (OUT / "conditions.json").write_text(json.dumps(cond, ensure_ascii=False, indent=1))
    pd.set_option("display.width", 250)
    print("\n## 조건\n", json.dumps(cond, ensure_ascii=False))
    print("\n## 지연(ms)\n", lat.pivot_table(index=["model", "method"], columns="batch", values=["mean_ms", "p95_ms"]).round(3).to_string())
    print("\n## 결과 차이\n", dif.to_string(index=False))
    print("\n## 모델 정보\n", inf.to_string(index=False))
    print("\n## 전력·GPU (배치 4)\n", lat[lat.batch == 4][["model", "method", "VDD_IN_mW", "VDD_CPU_GPU_CV_mW", "GPU_util_%", "RAM_used_MB"]].to_string(index=False))


if __name__ == "__main__":
    main()
