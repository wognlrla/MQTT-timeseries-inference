"""과제 3: GRU 60분 예측 모델 학습.

- 학습: 학습 기간, 입력 2시간+출력 60분 안에 고장이 없는 시점만
- 모델 선택(조기 종료): 검증 기간 손실. 시험 기간은 여기서 전혀 쓰지 않는다.
- 저장: models/<name>_s<seed>.pt (가중치+설정), preds/<name>_s<seed>.csv.gz (검증·시험 전 시점 예측)

사용: /srv/team/venv/bin/python train.py --name gru_sched --schedule 1 --seeds 0 1 2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, "/srv/team/common")
sys.path.insert(0, "/srv/team/task3")
import features as F  # noqa: E402
import preprocess as pp  # noqa: E402

ROOT = Path("/srv/team/task3")
STEPS = {5: 0, 15: 2, 30: 5, 60: 11}  # 예측 시점(분) → 출력 칸 번호


class GRUForecaster(nn.Module):
    def __init__(self, f_past, f_static, f_future, hidden=64, layers=2, horizon=F.HORIZON, n_out=2):
        super().__init__()
        self.horizon, self.n_out = horizon, n_out
        self.gru = nn.GRU(f_past, hidden, layers, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(hidden + f_static + f_future * horizon, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, horizon * n_out))

    def forward(self, past, static, future):
        _, h = self.gru(past)
        z = torch.cat([h[-1], static, future.flatten(1)], dim=1)
        return self.head(z).view(-1, self.horizon, self.n_out)


def masked_l1(pred, y, m):
    return (torch.abs(pred - y) * m).sum() / m.sum().clamp(min=1)


def to_t(arr, sel, dev):
    return [torch.from_numpy(arr[k][sel]).to(dev) for k in ("past", "static", "future")]


def run(args, seed, arr, y, ym, dev):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ok = arr["cur_ok"] & ~arr["fault_in_window"]
    tr = np.where((arr["split"] == "train") & ok)[0]
    va = np.where((arr["split"] == "val") & ok)[0]
    model = GRUForecaster(arr["past"].shape[2], arr["static"].shape[1], arr["future"].shape[2]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xv, yv, mv = to_t(arr, va, dev), torch.from_numpy(y[va]).to(dev), torch.from_numpy(ym[va]).to(dev)
    best, best_state, bad, log = 1e9, None, 0, []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            p = model(*to_t(arr, b, dev))
            loss = masked_l1(p, torch.from_numpy(y[b]).to(dev), torch.from_numpy(ym[b]).to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = masked_l1(model(*Xv), yv, mv).item()
        log.append(vl)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, {"best_val_l1": best, "epochs": len(log), "train_n": int(len(tr)), "val_n": int(len(va)),
                   "seconds": round(time.time() - t0, 1)}


def predict(model, arr, sel, dev, kind):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(sel), 4096):
            b = sel[i:i + 4096]
            out.append(model(*to_t(arr, b, dev)).cpu().numpy())
    z = np.concatenate(out)
    if kind == "delta":
        return arr["cur"][sel][:, None, :] + z * arr["std_t"]
    return arr["mean_t"] + z * arr["std_t"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--schedule", type=int, default=1)
    ap.add_argument("--target", default="delta", choices=["delta", "abs"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    (ROOT / "preds").mkdir(parents=True, exist_ok=True)

    grid, raw, _, schedule, faults, meta = pp.build()
    norm = json.loads(Path("/srv/team/task2/out/norm_stats.json").read_text())["stats"]
    arr = F.build_arrays(grid, schedule, meta, norm, use_schedule=bool(args.schedule))
    y, ym = F.targets(arr, args.target)
    sel = np.where(np.isin(arr["split"], ["val", "test"]) & arr["cur_ok"])[0]

    for seed in args.seeds:
        model, info = run(args, seed, arr, y, ym, dev)
        tag = f"{args.name}_s{seed}"
        torch.save({"state_dict": model.state_dict(), "config": vars(args) | {"seed": seed, "lookback": F.LOOKBACK,
                    "horizon": F.HORIZON, "targets": F.TARGETS, "norm": norm,
                    "dims": [arr["past"].shape[2], arr["static"].shape[1], arr["future"].shape[2]]}, "info": info},
                   ROOT / "models" / f"{tag}.pt")
        yhat = predict(model, arr, sel, dev, args.target)
        rows = []
        for h, k in STEPS.items():
            for j, v in enumerate(F.TARGETS):
                rows.append(pd.DataFrame({"time": arr["time"][sel], "space": arr["space"][sel], "var": v, "h": h,
                                          "yhat": yhat[:, k, j]}))
        pd.concat(rows).to_csv(ROOT / "preds" / f"{tag}.csv.gz", index=False)
        print(json.dumps({"model": tag, "device": dev} | info), flush=True)


if __name__ == "__main__":
    main()
