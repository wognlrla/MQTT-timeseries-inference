"""과제 2: 데이터 진단 (진단서 재료가 되는 표와 그래프).

1. 품질표: 공간·항목별 수신률, 결측률, 제거된 이상값 수, 5분 격자의 채움·긴 결측 비율
2. 긴 결측 구간 목록
3. 평일 시간대별 평균 CO₂·온도 그래프 (학습 기간, 고장 제외)
4. 일정 변화 뒤 CO₂ 반응: 수업·회의 시작 후 몇 분 만에 오르는가, 끝난 뒤 몇 분 만에 절반으로 내려가는가
5. 일정과 실제가 어긋난 일정: 예정이 있는데 CO₂가 안 오른 일정(휴강·노쇼), 예정 밖 CO₂ 상승(예약 없는 사용)
   — 합성 데이터의 정답(truth.csv.gz)과 대조해 판별이 맞았는지 확인한다(정답은 진단 검증에만 쓴다)

사용: /srv/team/venv/bin/python /srv/team/task2/diagnose.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "/srv/team/common")
import preprocess as pp  # noqa: E402

OUT = Path("/srv/team/task2/out")
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False
NAMES = {"classroom_a": "강의실 A", "classroom_b": "강의실 B", "meeting": "회의실", "lounge": "라운지"}
RISE_PPM = 30      # '반응했다'고 보는 CO₂ 상승 폭(잡음 σ 8ppm의 약 4배)
EMPTY_RISE = 40    # 일정 동안 이만큼도 안 오르면 '안 쓴 것 같다'
UNSCHED_RISE = 80  # 일정 밖 30분 동안 이만큼 오르면 '예정 밖 사용'


def quality(sensors, raw, grid, meta):
    rows = []
    minutes = (pd.Timestamp(meta["end"]) - pd.Timestamp(meta["start"])).total_seconds() / 60 + 1
    for space, cfg in meta["spaces"].items():
        expected = minutes / cfg["report_min"]
        s_raw = sensors[sensors.space == space]
        s_cln = raw[raw.space == space]
        g = grid[grid.space == space]
        for v in pp.VARS:
            lo, hi = pp.RANGES[v]
            bad = int((s_raw[v].notna() & ((s_raw[v] < lo) | (s_raw[v] > hi))).sum())
            rows.append({
                "공간": NAMES[space], "항목": v, "기록주기(분)": cfg["report_min"],
                "예상 기록 수": int(expected), "받은 기록 수": len(s_raw),
                "결측률(%)": round(100 * (1 - s_cln[v].notna().sum() / expected), 2),
                "제거한 이상값": bad,
                "5분칸 비어있음(%)": round(100 * (g["n_obs"] == 0).mean(), 2),
                "직전값 채움(%)": round(100 * g[f"{v}_filled"].mean(), 2),
                "긴 결측 남음(%)": round(100 * g[f"{v}_gap"].mean(), 2),
            })
    return pd.DataFrame(rows)


def gap_episodes(raw, meta, min_minutes=15):
    eps = []
    for space, cfg in meta["spaces"].items():
        t = raw[(raw.space == space) & raw["co2"].notna()]["time"].sort_values()
        d = t.diff().dt.total_seconds() / 60
        for end, dm in zip(t[d > max(min_minutes, cfg["report_min"] * 2)], d[d > max(min_minutes, cfg["report_min"] * 2)]):
            eps.append({"공간": NAMES[space], "시작": end - pd.Timedelta(minutes=dm), "끝": end,
                        "길이(분)": int(dm)})
    return pd.DataFrame(eps).sort_values("길이(분)", ascending=False)


def weekday_profile(grid, meta):
    a, b = pp.split_bounds(meta)["train"]
    g = grid[(grid.time >= a) & (grid.time < b) & ~grid.fault & (grid.time.dt.weekday < 5)].copy()
    g["tod"] = g.time.dt.hour + g.time.dt.minute / 60
    prof = g.groupby(["space", "tod"])[["co2", "temperature"]].mean().reset_index()
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for space, p in prof.groupby("space"):
        ax[0].plot(p.tod, p.co2, label=NAMES[space])
        ax[1].plot(p.tod, p.temperature, label=NAMES[space])
    for x, lab in [(9, "수업 시작"), (12, "점심"), (13, "오후 수업"), (18, "퇴실")]:
        for a_ in ax:
            a_.axvline(x, color="grey", lw=0.6, ls="--")
        ax[0].text(x + 0.1, ax[0].get_ylim()[1] * 0.97, lab, fontsize=8, va="top")
    ax[0].axhline(1000, color="red", lw=0.8)
    ax[0].set_ylabel("CO2 (ppm)")
    ax[1].set_ylabel("온도 (℃)")
    ax[1].set_xlabel("시각 (평일, 학습 기간 평균)")
    ax[1].set_xticks(range(0, 25, 2))
    ax[0].legend(ncol=4, fontsize=9, loc="upper left", bbox_to_anchor=(0, -0.02, 1, 0), mode="expand", frameon=False)
    ax[0].set_title("평일 시간대별 평균 CO2·온도 (5분 격자, 고장 제외)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_weekday_profile.png", dpi=130)
    plt.close(fig)
    peak = prof.loc[prof.groupby("space").co2.idxmax(), ["space", "tod", "co2"]]
    return prof, peak


def schedule_response(raw, schedule, faults, meta):
    """1분 원본으로 일정 시작·끝 전후 CO₂를 맞춰 본다(학습 기간, 고장 일정 제외)."""
    a, b = pp.split_bounds(meta)["train"]
    ev = schedule[schedule.kind.isin(["class", "booking"]) & (schedule.start >= a) & (schedule.end < b)]
    rows, curves = [], []
    for space, e in ev.groupby("space"):
        s = raw[raw.space == space].set_index("time")["co2"].dropna()
        s = s.resample("1min").mean().interpolate(limit=5)
        for r in e.itertuples():
            base = s.get(r.start)
            if base is None or np.isnan(base):
                continue
            win = s[r.start: r.start + pd.Timedelta(minutes=60)] - base
            hit = win[win >= RISE_PPM]
            rise_min = (hit.index[0] - r.start).total_seconds() / 60 if len(hit) else np.nan
            during = s[r.start: r.end]
            peak = during.max()
            after = s[r.end: r.end + pd.Timedelta(minutes=120)]
            half = base + (peak - base) / 2
            fall = after[after <= half]
            fall_min = (fall.index[0] - r.end).total_seconds() / 60 if len(fall) and peak - base >= RISE_PPM else np.nan
            rows.append({"space": space, "kind": r.kind, "start": r.start, "end": r.end, "expected": r.expected,
                         "base": base, "rise_ppm": peak - base, "react_min": rise_min, "half_decay_min": fall_min})
            rel = s[r.start - pd.Timedelta(minutes=10): r.start + pd.Timedelta(minutes=60)] - base
            curves.append(pd.DataFrame({"space": space, "m": (rel.index - r.start).total_seconds() / 60, "d": rel.values}))
    ev = pd.DataFrame(rows)
    used = ev[ev.rise_ppm >= EMPTY_RISE]
    summary = used.groupby("space").agg(
        일정수=("start", "size"), 반응_중앙값_분=("react_min", "median"),
        반응_25_75=("react_min", lambda x: f"{x.quantile(.25):.0f}~{x.quantile(.75):.0f}"),
        상승폭_중앙값_ppm=("rise_ppm", "median"), 반감_중앙값_분=("half_decay_min", "median")).round(1)
    summary.index = summary.index.map(NAMES)
    cv = pd.concat(curves)
    cv = cv[cv.m.between(-10, 60)]
    fig, ax = plt.subplots(figsize=(8, 4))
    for space, c in cv.groupby("space"):
        m = c.groupby(c.m.round()).d.median()
        ax.plot(m.index, m.values, label=NAMES[space])
    ax.axvline(0, color="grey", ls="--", lw=0.8)
    ax.axhline(RISE_PPM, color="red", lw=0.6, ls=":")
    ax.set_xlabel("일정 시작 기준 경과 시간 (분)")
    ax.set_ylabel("시작 시점 대비 CO2 변화 (ppm, 중앙값)")
    ax.set_title("수업·회의 시작 뒤 CO2 반응 (학습 기간)")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_schedule_response.png", dpi=130)
    plt.close(fig)
    return ev, summary


def mismatches(ev_all, raw, schedule, truth, meta):
    """일정이 있는데 안 쓴 것 같은 일정 + 일정 밖 사용. 정답과 대조한다."""
    tr = truth.set_index(["space", "time"])["occupants"]
    out = []
    for r in ev_all.itertuples():
        occ = tr.loc[r.space].loc[r.start: r.end]
        out.append({"공간": NAMES[r.space], "종류": r.kind, "시작": r.start, "끝": r.end,
                    "CO2상승": round(r.rise_ppm, 0), "판별_안씀": r.rise_ppm < EMPTY_RISE,
                    "정답_평균인원": round(occ.mean(), 1)})
    sched = pd.DataFrame(out)
    sched["정답_안씀"] = sched["정답_평균인원"] < 0.5
    conf = pd.crosstab(sched["판별_안씀"], sched["정답_안씀"], rownames=["CO₂ 판별: 안 씀"], colnames=["정답: 안 씀"])

    # 일정 밖 사용: 평일 8~20시, 어떤 일정에도 걸치지 않는 30분 창에서 CO₂가 크게 오름
    unsched = []
    for space in ["classroom_a", "classroom_b", "meeting"]:
        s = raw[raw.space == space].set_index("time")["co2"].dropna().resample("5min").mean()
        busy = pd.Series(False, index=s.index)
        for r in schedule[schedule.space == space].itertuples():
            busy[(s.index >= r.start - pd.Timedelta(minutes=30)) & (s.index <= r.end + pd.Timedelta(minutes=30))] = True
        rise = s - s.shift(6)
        cand = rise[(rise >= UNSCHED_RISE) & ~busy & (s.index.hour.isin(range(8, 20)))]
        for day, c in cand.groupby(cand.index.date):
            t = c.idxmax()
            occ = tr.loc[space].loc[t - pd.Timedelta(minutes=30): t]
            unsched.append({"공간": NAMES[space], "날짜": str(day), "시각": t.strftime("%H:%M"),
                            "30분상승": round(c.max(), 0), "정답_평균인원": round(occ.mean(), 1)})
    return sched, conf, pd.DataFrame(unsched)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    grid, raw, raw_stats, schedule, faults, meta = pp.build()
    sensors, *_ = pp.load_synthetic()
    truth = pd.read_csv("/srv/team/data/synthetic/truth.csv.gz", parse_dates=["time"])

    q = quality(sensors, raw, grid, meta)
    q.to_csv(OUT / "diag_quality.csv", index=False)
    gaps = gap_episodes(raw, meta)
    gaps.to_csv(OUT / "diag_gaps.csv", index=False)
    prof, peak = weekday_profile(grid, meta)
    ev, resp = schedule_response(raw, schedule, faults, meta)
    resp.to_csv(OUT / "diag_schedule_response.csv")
    sched, conf, unsched = mismatches(ev, raw, schedule, truth, meta)
    sched[sched["판별_안씀"] | sched["정답_안씀"]].to_csv(OUT / "diag_schedule_mismatch.csv", index=False)
    conf.to_csv(OUT / "diag_mismatch_confusion.csv")
    unsched.to_csv(OUT / "diag_unscheduled_use.csv", index=False)

    pd.set_option("display.width", 220)
    print("## 품질\n", q.to_string(index=False))
    print("\n## 긴 결측 상위 10\n", gaps.head(10).to_string(index=False), f"\n(총 {len(gaps)}건)")
    print("\n## 평일 CO₂ 최고 시각\n", peak.assign(space=peak.space.map(NAMES)).round(1).to_string(index=False))
    print("\n## 일정 시작 후 반응\n", resp.to_string())
    print("\n## 일정 대비 미사용 판별 vs 정답\n", conf.to_string())
    print("\n## 안 쓴 것으로 보이는 일정 (판별 또는 정답)\n",
          sched[sched["판별_안씀"] | sched["정답_안씀"]].to_string(index=False))
    print("\n## 일정 밖 사용 후보\n", unsched.to_string(index=False))
    (OUT / "diag_summary.json").write_text(json.dumps({"raw_cleaning": raw_stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
