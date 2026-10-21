"""과제 6: 모델 하나를 새 프로세스에서 올려 '서비스 1회 추론' 비용을 잰다.

사용: python cost.py <chronos-bolt-tiny|small|base|gru> <ctx>
출력: JSON 한 줄 — 프로세스 메모리 증가(MB), GPU 최대 할당(MB), 배치 1·4 지연(평균·p95 ms, GPU/CPU)
"""
import json
import sys
import time

import numpy as np
import torch

name, L = sys.argv[1], int(sys.argv[2])
H, Q = 12, [0.1, 0.5, 0.9]


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024


def latency(fn, dev, n=100, warm=10):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if dev == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return round(float(np.mean(ts)), 2), round(float(np.percentile(ts, 95)), 2)


torch.zeros(1).cuda()  # CUDA 초기화 비용은 공통이므로 기준선에 포함
torch.cuda.synchronize()
base_rss = rss_mb()
torch.cuda.reset_peak_memory_stats()
res = {"model": name, "ctx": L}
if name == "gru":
    sys.path.insert(0, "/srv/team/task3")
    from train import GRUForecaster
    ck = torch.load("/srv/team/task3/models/gru_sched_s0.pt", weights_only=False)
    fp, fs, ff = ck["config"]["dims"]
    for dev in ("cuda", "cpu"):
        m = GRUForecaster(fp, fs, ff).to(dev).eval()
        m.load_state_dict(ck["state_dict"])
        res["params(M)"] = round(sum(p.numel() for p in m.parameters()) / 1e6, 3)
        for b in (1, 4):
            x = (torch.randn(b, 24, fp, device=dev), torch.randn(b, fs, device=dev), torch.randn(b, H, ff, device=dev))
            with torch.no_grad():
                res[f"{dev}_b{b}"] = latency(lambda: m(*x), dev)
else:
    from chronos import BaseChronosPipeline
    for dev in ("cuda", "cpu"):
        pipe = BaseChronosPipeline.from_pretrained(f"amazon/{name}", device_map=dev, torch_dtype=torch.float32)
        res["params(M)"] = round(sum(p.numel() for p in pipe.model.parameters()) / 1e6, 1)
        for b in (1, 4):
            x = torch.randn(b, L) * 100 + 600
            res[f"{dev}_b{b}"] = latency(lambda: pipe.predict_quantiles(x, prediction_length=H, quantile_levels=Q), dev,
                                         n=100 if dev == "cuda" else 20, warm=10 if dev == "cuda" else 3)
        if dev == "cuda":
            res["gpu_peak_MB"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            res["rss_increase_MB"] = round(rss_mb() - base_rss)
        del pipe
if name == "gru":
    res["gpu_peak_MB"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    res["rss_increase_MB"] = round(rss_mb() - base_rss)
print(json.dumps(res, ensure_ascii=False))
