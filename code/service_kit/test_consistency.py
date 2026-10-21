"""학습-서비스 전처리 일치 시험.

시험 기간에서 무작위 시점 N개를 골라,
- 학습 때 함수(task3 build_arrays / task4 ae_windows / task5 build_table)가 전체 데이터로 만든 입력과
- 키트가 서비스처럼 '최근 4시간 원본만' 보고 만든 입력을
항목별로 비교한다. 차이가 0이면 서비스는 학습 때와 똑같은 입력을 모델에 넣는다.
마지막으로 한 시점은 키트 전체(run)를 돌려 과제 3 예측과 출력도 비교한다.

사용: /srv/team/venv/bin/python test_consistency.py [N=40]
"""
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/srv/team/service_kit")
import kit as K  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    kit = K.Kit()
    grid, raw, _, schedule, faults, meta = K.pp.build()
    sensors, *_ = K.pp.load_synthetic()
    norm = kit.norm
    arr = K.F.build_arrays(grid, schedule, meta, norm, use_schedule=True)
    w = K.D.ae_windows(grid, schedule, meta, norm)
    t5 = load("t5run", "/srv/team/task5/run.py")
    tab, _, _ = t5.build_table()

    a, b = K.pp.split_bounds(meta)["test"]
    rng = np.random.default_rng(0)
    cands = pd.date_range(a + pd.Timedelta(hours=5), b, freq="5min")
    times = sorted(rng.choice(cands, N, replace=False))
    worst = {"past": 0, "static": 0, "future": 0, "ae_x": 0, "ae_c": 0, "occ": 0, "vent": 0}
    checked = 0
    for T in times:
        T = pd.Timestamp(T)
        rw = sensors[(sensors.time >= T - pd.Timedelta(hours=K.WINDOW_H)) & (sensors.time < T)]
        prep, _, _ = kit.prepare(rw, schedule, T + pd.Timedelta(seconds=30))
        for s in K.SPACES:
            d = prep[s]
            i = np.where((arr["time"] == np.datetime64(T)) & (arr["space"] == s))[0]
            j = np.where((w["time"] == np.datetime64(T)) & (w["space"] == s))[0]
            if not len(i) or not arr["cur_ok"][i[0]]:
                continue
            i, j = i[0], j[0]
            worst["past"] = max(worst["past"], float(np.abs(arr["past"][i] - d["past"]).max()))
            worst["static"] = max(worst["static"], float(np.abs(arr["static"][i] - d["static"]).max()))
            worst["future"] = max(worst["future"], float(np.abs(arr["future"][i] - d["future"]).max()))
            if w["ok"][j] and d["ae_ok"]:
                worst["ae_x"] = max(worst["ae_x"], float(np.abs(w["x"][j] - d["ae_x"]).max()))
                worst["ae_c"] = max(worst["ae_c"], float(np.abs(w["c"][j] - d["ae_c"]).max()))
            row = tab[(tab.time == T) & (tab.space == s)]
            if len(row) and not np.isnan(d["grid"].co2.iloc[-13:]).any():
                feats = kit.state_features(s, d)
                for key, ck in (("occ", kit.occ), ("vent", kit.vent)):
                    ref = ((row[ck["cols"]].iloc[0] - pd.Series(ck["norm"]["mean"])[ck["cols"]])
                           / pd.Series(ck["norm"]["std"])[ck["cols"]]).fillna(0).to_numpy(np.float32)
                    worst[key] = max(worst[key], float(np.abs(ref - kit._mlp_input(ck, feats)[0]).max()))
            checked += 1

    # 키트 전체 실행 1회: 과제 3 예측(PyTorch)과 비교
    T = pd.Timestamp("2026-10-07 10:30")
    rw = sensors[(sensors.time >= T - pd.Timedelta(hours=K.WINDOW_H)) & (sensors.time < T + pd.Timedelta(seconds=30))]
    res = kit.run(rw, schedule, T + pd.Timedelta(seconds=30))
    p3 = pd.read_csv("/srv/team/task3/out/all_predictions.csv.gz", parse_dates=["time"])
    p3 = p3[(p3.model == "gru_sched") & (p3.seed == 0) & (p3.time == T) & (p3["var"] == "co2")]
    cmp = []
    for r in p3.itertuples():
        f = res[r.space]["forecast"]
        if f["available"]:
            cmp.append(abs(f["co2"][f["minutes_ahead"].index(r.h)] - r.yhat))
    out = {"checked_space_times": checked, "max_abs_diff_inputs": {k: round(v, 7) for k, v in worst.items()},
           "forecast_vs_task3_pytorch_max_ppm": round(max(cmp), 3) if cmp else None,
           "backends": kit.backends()}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(res["meeting"], ensure_ascii=False, indent=1)[:2500])


if __name__ == "__main__":
    main()
