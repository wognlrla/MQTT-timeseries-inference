"""과제 8 추론 키트: 최근 센서 기록 → 공간별 판단 결과(예측·이상·상태) 한 번에.

학습과 서비스의 전처리를 같게 하려고, 과제 2~5에서 쓴 함수를 그대로 불러다 쓴다.
- 정제·5분 격자: common/preprocess.py (clean_raw, to_grid)
- 예측 입력: task3/features.py (schedule_grid, static_feats) — build_arrays와 같은 순서·정규화
- 이상 탐지 규칙·점수: task4/detectors.py, task4/out/summary.json의 기준값
- 상태 추정 입력: task5/run.py build_table과 같은 계산식
모델 실행은 TensorRT(과제 7 엔진)를 먼저 쓰고, 실패하면 ONNX Runtime(CPU, 스레드 1개)으로 넘어간다.

사용:
    kit = Kit()
    results = kit.run(raw, schedule, now)   # raw: time(KST, naive), space, co2, temperature, humidity
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

for p in ("/srv/team/common", "/srv/team/task3", "/srv/team/task4"):
    if p not in sys.path:
        sys.path.insert(0, p)
import detectors as D  # noqa: E402
import features as F  # noqa: E402
import preprocess as pp  # noqa: E402

TEAM = Path("/srv/team")
ENG = TEAM / "task7" / "engines"
SPACES = F.SPACES
NAMES = {"classroom_a": "강의실 A", "classroom_b": "강의실 B", "meeting": "회의실", "lounge": "라운지"}
LIMIT = 1000.0
HORIZON_MIN = [5 * (k + 1) for k in range(F.HORIZON)]
WINDOW_H = 4                 # 서비스가 읽어 오는 과거 길이(시간). 입력 2시간 + 결측 채움 여유
SIM_WEEK_START = pd.Timestamp("2026-10-05")   # 합성 시험 주간 월요일 (simfeed.py와 같아야 함)


# ------------------------------------------------------------------ 모델 실행기
class _TRT:
    def __init__(self, path):
        import tensorrt as trt
        import torch
        self.trt, self.torch = trt, torch
        log = trt.Logger(trt.Logger.ERROR)
        self.rt = trt.Runtime(log)
        self.eng = self.rt.deserialize_cuda_engine(Path(path).read_bytes())
        if self.eng is None:
            raise RuntimeError(f"엔진을 읽지 못함: {path}")
        self.ctx = self.eng.create_execution_context()
        names = [self.eng.get_tensor_name(i) for i in range(self.eng.num_io_tensors)]
        self.ins = [n for n in names if self.eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.out = [n for n in names if self.eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT][0]
        self.stream = torch.cuda.Stream()

    def __call__(self, *arrs):
        torch = self.torch
        with torch.cuda.stream(self.stream):
            ts = [torch.from_numpy(np.ascontiguousarray(a, np.float32)).cuda(non_blocking=True) for a in arrs]
            for n, t in zip(self.ins, ts):
                self.ctx.set_input_shape(n, tuple(t.shape))
                self.ctx.set_tensor_address(n, t.data_ptr())
            o = torch.empty(tuple(self.ctx.get_tensor_shape(self.out)), dtype=torch.float32, device="cuda")
            self.ctx.set_tensor_address(self.out, o.data_ptr())
            if not self.ctx.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT 실행 실패")
            res = o.cpu()
        self.stream.synchronize()
        return res.numpy()


class Model:
    """TensorRT 우선, 실패 시 ONNX Runtime(CPU)."""
    def __init__(self, onnx_name, engine_name, input_names, prefer="tensorrt"):
        self.onnx = ENG / onnx_name
        self.names = input_names
        self.backend, self.error, self._ort = None, None, None
        if prefer == "tensorrt":
            try:
                self._trt = _TRT(ENG / engine_name)
                self.backend = "tensorrt"
            except Exception as e:  # GPU가 없거나 엔진이 맞지 않음
                self.error = f"{type(e).__name__}: {e}"
        if self.backend is None:
            self._use_ort()

    def _use_ort(self):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self._ort = ort.InferenceSession(str(self.onnx), so, providers=["CPUExecutionProvider"])
        self.backend = "onnxruntime"

    def __call__(self, *arrs):
        if self.backend == "tensorrt":
            try:
                return self._trt(*arrs)
            except Exception as e:
                self.error = f"{type(e).__name__}: {e} → ONNX Runtime으로 전환"
                self._use_ort()
        return self._ort.run(None, {n: np.ascontiguousarray(a, np.float32) for n, a in zip(self.names, arrs)})[0]


# ------------------------------------------------------------------ 합성 재생 시간 매핑
def week_start(t):
    return (t - pd.Timedelta(days=t.weekday())).normalize()


def to_sim_time(t):
    """현재 시각 → 같은 요일·시각의 합성 시험 주간 시각."""
    return SIM_WEEK_START + (t - week_start(t))


def shifted_schedule(schedule, now):
    """합성 시험 주간의 일정을 지금 주(와 앞뒤 주)로 옮긴다. simfeed의 센서 재생과 같은 규칙."""
    sim = schedule[(schedule.start >= SIM_WEEK_START) & (schedule.start < SIM_WEEK_START + pd.Timedelta(days=7))]
    parts = []
    for k in (-1, 0, 1):
        off = week_start(now) + pd.Timedelta(days=7 * k) - SIM_WEEK_START
        parts.append(sim.assign(start=sim.start + off, end=sim.end + off))
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------ 키트
class Kit:
    def __init__(self, prefer="tensorrt"):
        import torch  # .pt 설정 파일 읽기용
        self.meta = json.loads((TEAM / "data/synthetic/meta.json").read_text())
        self.norm = json.loads((TEAM / "task2/out/norm_stats.json").read_text())["stats"]
        self.mean = np.array([self.norm[v]["mean"] for v in pp.VARS], np.float32)
        self.std = np.array([self.norm[v]["std"] for v in pp.VARS], np.float32)
        s4 = json.loads((TEAM / "task4/out/summary.json").read_text())
        self.ae_thr = float(s4["ae_threshold"])
        self.spike_thr = {tuple(k.split("/")): v for k, v in s4["spike_thresholds"].items()}
        self.level_thr, self.night_thr = s4["level_thresholds"], s4["night_thresholds"]
        ae = torch.load(TEAM / "task4/ae.pt", weights_only=False)
        self.ae_scale = np.array(ae["scale"], np.float32)
        self.occ = torch.load(TEAM / "task5/occ_mlp.pt", weights_only=False)
        self.vent = torch.load(TEAM / "task5/vent_mlp.pt", weights_only=False)
        self.m_fc = Model("forecast_gru.onnx", "forecast_gru.fp16_headFP32.engine", ["past", "static", "future"], prefer)
        self.m_ae = Model("anomaly_ae.onnx", "anomaly_ae.fp16.engine", ["x", "c"], prefer)
        self.m_occ = Model("occupancy_mlp.onnx", "occupancy_mlp.fp16.engine", ["x"], prefer)
        self.m_vent = Model("vent_mlp.onnx", "vent_mlp.fp16.engine", ["x"], prefer)

    def backends(self):
        return {k: {"backend": m.backend, "error": m.error} for k, m in
                [("forecast", self.m_fc), ("anomaly", self.m_ae), ("occupancy", self.m_occ), ("vent", self.m_vent)]}

    # -------------------------------------------------------------- 입력 만들기
    def prepare(self, raw, schedule, now):
        """학습과 같은 규칙으로 5분 격자와 모델 입력을 만든다. 마지막 칸 = 이미 끝난 5분 칸."""
        t_end = pd.Timestamp(now).floor(pp.FREQ)            # 막 시작된 칸(1~2분치)은 쓰지 않는다
        raw = raw[(raw.time < t_end) & (raw.time >= t_end - pd.Timedelta(hours=WINDOW_H))]
        clean, stats = pp.clean_raw(raw)
        grid = pp.to_grid(clean, t_end - pd.Timedelta(hours=WINDOW_H), t_end - pd.Timedelta(pp.FREQ))
        idx = pd.DatetimeIndex(sorted(grid.time.unique()))
        idx_ext = idx.append(pd.date_range(idx[-1] + pd.Timedelta(pp.FREQ), periods=F.HORIZON, freq=pp.FREQ))
        sched = F.schedule_grid(schedule, idx_ext, self.meta)
        out = {}
        for s in SPACES:
            g = grid[grid.space == s].set_index("time").reindex(idx)
            rawv = g[pp.VARS].to_numpy(np.float32)
            miss = np.isnan(rawv)
            z = np.where(miss, 0.0, (rawv - self.mean) / self.std).astype(np.float32)
            sc = sched[s].to_numpy(np.float32)
            L, T = F.LOOKBACK, len(idx)
            past = np.concatenate([z, miss.astype(np.float32), sc[:T]], axis=1)[T - L:]
            out[s] = dict(grid=g, t=idx[-1], cur=rawv[-1, F.T_IDX], past=past, future=sc[T:T + F.HORIZON],
                          static=F.static_feats(pd.DatetimeIndex([idx[-1]]), s)[0],
                          ae_x=z[T - D.WIN:].reshape(-1), ae_c=np.concatenate(
                              [F.static_feats(pd.DatetimeIndex([idx[-1]]), s)[0], sc[T - D.WIN:T].reshape(-1)]),
                          ae_ok=not miss[T - D.WIN:].any(), sched=sched[s])
        return out, clean, stats

    def state_features(self, s, d):
        """과제 5 build_table과 같은 계산식으로 마지막 칸의 특징."""
        g, sc = d["grid"], d["sched"]
        co2, tmp, hum = g.co2, g.temperature, g.humidity
        t = d["t"]
        h = t.hour + t.minute / 60
        f = {"co2": co2.iloc[-1], "co2_ex": co2.iloc[-1] - 420.0, "temp": tmp.iloc[-1],
             "co2_rng30": co2.rolling(7).max().iloc[-1] - co2.rolling(7).min().iloc[-1],
             "co2_max60": co2.rolling(13).max().iloc[-1],
             "h_sin": np.sin(2 * np.pi * h / 24), "h_cos": np.cos(2 * np.pi * h / 24),
             "dow_sin": np.sin(2 * np.pi * t.weekday() / 7), "dow_cos": np.cos(2 * np.pi * t.weekday() / 7),
             "weekend": float(t.weekday() >= 5)}
        for k, n in [("5", 1), ("15", 3), ("30", 6)]:
            f[f"dco2_{k}"] = co2.diff(n).iloc[-1]
            f[f"dtemp_{k}"] = tmp.diff(n).iloc[-1]
            f[f"dhum_{k}"] = hum.diff(n).iloc[-1]
        f["sched_on"], f["sched_exp"] = sc.sched_on.loc[t], sc.sched_exp.loc[t]
        nxt = sc.loc[sc.index > t].iloc[:6]
        f["sched_on_next30"], f["sched_exp_next30"] = nxt.sched_on.max(), nxt.sched_exp.max()
        for x in SPACES:
            f[f"sp_{x}"] = float(x == s)
        return f

    def _mlp_input(self, ckpt, feats):
        mean, std = pd.Series(ckpt["norm"]["mean"]), pd.Series(ckpt["norm"]["std"])
        row = pd.Series({c: feats.get(c, np.nan) for c in ckpt["cols"]}, dtype=float)
        return ((row - mean[ckpt["cols"]]) / std[ckpt["cols"]]).fillna(0).to_numpy(np.float32)[None, :]

    # -------------------------------------------------------------- 규칙
    def rules(self, s, raw_all, clean, d, now):
        rep = self.meta["spaces"][s]["report_min"]
        r = {}
        mine = raw_all[raw_all.space == s]
        last = mine.time.max() if len(mine) else None
        age = (now - last).total_seconds() / 60 if last is not None else None
        gap_thr = max(10, 3 * rep)
        r["data_gap"] = {"active": age is None or age > gap_thr, "last_seen": None if last is None else last.isoformat(),
                         "age_min": None if age is None else round(age, 1), "threshold_min": gap_thr}
        recent = mine[mine.time > now - pd.Timedelta(minutes=10)]
        bad = {v: int((recent[v].notna() & ((recent[v] < lo) | (recent[v] > hi))).sum()) for v, (lo, hi) in pp.RANGES.items()}
        r["out_of_range"] = {"active": sum(bad.values()) > 0, "count_10min": bad}
        cs = clean[clean.space == s]
        st = D.stuck_alarms(cs) if len(cs) else pd.DataFrame(columns=["time"])
        r["stuck"] = {"active": bool(len(st) and st.time.max() >= now - pd.Timedelta(minutes=2 * rep + 5)),
                      "minutes": None if not len(st) else round(float(st.score.max() * D.STUCK_MIN), 0)}
        sp = D.spike_alarms(cs, {k: v for k, v in self.spike_thr.items() if k[0] == s}) if len(cs) > 6 else pd.DataFrame(columns=["time", "var"])
        sp = sp[sp.time >= now - pd.Timedelta(minutes=10)]
        r["spike"] = {"active": bool(len(sp)), "vars": sorted(set(sp["var"]))}
        co2 = d["grid"].co2
        r["co2_level"] = {"active": bool(co2.iloc[-1] > self.level_thr[s]), "threshold_ppm": round(self.level_thr[s])}
        tail = co2.iloc[-D.NIGHT_MIN_CELLS:]
        r["night_baseline"] = {"active": bool(d["t"].hour in D.NIGHT_HOURS and len(tail) and (tail > self.night_thr[s]).all()),
                               "threshold_ppm": round(self.night_thr[s])}
        return r

    # -------------------------------------------------------------- 실행
    def run(self, raw, schedule, now):
        """raw: 최근 WINDOW_H시간 이상의 원본 기록(time KST naive, space, co2, temperature, humidity)."""
        now = pd.Timestamp(now)
        prep, clean, stats = self.prepare(raw, schedule, now)
        res = {}
        ok_fc = [s for s in SPACES if not np.isnan(prep[s]["cur"]).any()]
        if ok_fc:
            z = self.m_fc(np.stack([prep[s]["past"] for s in ok_fc]), np.stack([prep[s]["static"] for s in ok_fc]),
                          np.stack([prep[s]["future"] for s in ok_fc]))
            std_t = self.std[F.T_IDX]
            fc = {s: prep[s]["cur"] + z[i] * std_t for i, s in enumerate(ok_fc)}  # 변화량 → 원래 단위
        ok_ae = [s for s in SPACES if prep[s]["ae_ok"]]
        if ok_ae:
            x = np.stack([prep[s]["ae_x"] for s in ok_ae])
            rec = self.m_ae(x, np.stack([prep[s]["ae_c"] for s in ok_ae]))
            err = ((rec - x) ** 2).reshape(len(x), D.WIN, len(pp.VARS))[:, -D.SCORE_LAST:, :].mean(1) / self.ae_scale
        for s in SPACES:
            d = prep[s]
            g = d["grid"]
            item = {"space": s, "name": NAMES[s], "time": d["t"].isoformat(), "computed_at": now.isoformat(),
                    "now": {"co2": _f(g.co2.iloc[-1]), "temperature": _f(g.temperature.iloc[-1], 2),
                            "humidity": _f(g.humidity.iloc[-1], 1),
                            "filled_cells_2h": int(sum(g[f"{v}_filled"].iloc[-24:].sum() for v in pp.VARS)),
                            "missing_cells_2h": int(g.co2.iloc[-24:].isna().sum())}}
            if s in ok_fc:
                co2f, tf = fc[s][:, 0], fc[s][:, 1]
                cur = float(d["cur"][0])
                over = [m for m, v in zip(HORIZON_MIN, co2f) if v >= LIMIT]
                item["forecast"] = {"available": True, "minutes_ahead": HORIZON_MIN,
                                    "co2": [round(float(v), 1) for v in co2f], "temperature": [round(float(v), 2) for v in tf],
                                    "already_over_1000": cur >= LIMIT,
                                    "exceed_1000_in_min": None if cur >= LIMIT or not over else over[0]}
            else:
                item["forecast"] = {"available": False, "reason": "현재 CO2·온도 값 없음(긴 결측)"}
            rules = self.rules(s, raw, clean, d, now)
            an = {"available": s in ok_ae, "threshold": self.ae_thr, "rules": rules}
            if s in ok_ae:
                e = err[ok_ae.index(s)]
                order = np.argsort(-e)
                an.update(score=round(float(e.max()), 3), is_anomaly=bool(e.max() > self.ae_thr),
                          cause=[{"var": pp.VARS[i], "z": round(float(e[i]), 3)} for i in order[:2]])
            else:
                an.update(score=None, is_anomaly=False, reason="최근 2시간에 결측 칸이 있어 점수 계산 안 함")
            an["any_alert"] = bool(an.get("is_anomaly") or any(v["active"] for v in rules.values()))
            item["anomaly"] = an
            if not np.isnan(g.co2.iloc[-13:]).any():
                feats = self.state_features(s, d)
                p_occ = float(1 / (1 + np.exp(-self.m_occ(self._mlp_input(self.occ, feats))[0])))
                p_vent = float(1 / (1 + np.exp(-self.m_vent(self._mlp_input(self.vent, feats))[0])))
                fc_alert = bool(s in ok_fc and not item["forecast"]["already_over_1000"]
                                and item["forecast"]["exceed_1000_in_min"] is not None and item["forecast"]["exceed_1000_in_min"] <= 30)
                item["state"] = {"available": True, "occupied": p_occ >= 0.5, "occupied_prob": round(p_occ, 3),
                                 "vent_alert": fc_alert, "vent_alert_source": "forecast(과제 5 결론)",
                                 "vent_prob_classifier": round(p_vent, 3),
                                 "vent_alert_classifier": bool(feats["co2"] < LIMIT and p_vent >= self.vent["threshold"])}
            else:
                item["state"] = {"available": False, "reason": "최근 1시간 CO2 결측"}
            res[s] = item
        return res


def _f(v, nd=0):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), nd)
