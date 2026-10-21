# team2 교육장 추론 서비스 v1 — 결과 형식

공간 4곳(강의실 A·B, 회의실, 라운지)의 **60분 예측, 이상 탐지, 사용 여부·환기 선제 경보**를 **매분 15초**에 새로 계산해 내보낸다.
다른 팀은 이 문서만 보고 MQTT로 받거나 HTTP로 물어보면 된다.

| 항목 | 값 |
|---|---|
| 보드 | team2 Jetson (Tailscale `<JETSON_HOST>`, 교육장 내부망에서는 보드 IP) |
| MQTT | 포트 **1885**, 토픽 접두어 **`team2/svc/`**, **로그인 없이 구독 가능**(읽기 전용) |
| HTTP API | **`http://<보드>:8091`**, 대화형 문서 `http://<보드>:8091/docs` |
| 갱신 주기 | 매분 15초 (예: 09:23:15). 결과 시각(`time`)은 그 직전에 **끝난** 5분 칸 |
| 입력 데이터 | 합성 재생 데이터(실데이터 3주가 쌓이기 전 과제 규정). 시험 주간(10-05 월 ~ 10-11 일)을 현재 요일·시각에 맞춰 재생한다 |
| 시각 | 한국 시간(KST), 시간대 표기 없는 ISO 8601 (`2026-10-21T09:20:00`) |
| 단위 | CO₂ **ppm**, 온도 **℃**, 습도 **%**, 시간 **분** |

---

## 1. MQTT 토픽

| 토픽 | retain | 내용 |
|---|---|---|
| `team2/svc/state/<space>` | ✅ | 공간별 전체 결과 (아래 2절). 새로 구독하면 즉시 최신값을 받는다 |
| `team2/svc/alerts/<space>` | ✅ | 지금 켜져 있는 경보 이름 목록 |
| `team2/svc/events` | ❌ | 경보 **변화**(새로 울림·30분마다 재알림·해제)만 흐른다 |
| `team2/svc/status` | ✅ | 서비스 상태. 서비스가 비정상 종료되면 브로커가 `{"online": false}`로 바꾼다(LWT) |
| `team2/svc/real/<location>` | ✅ | 실데이터 공간(실습실·사무실·사무실 밖·회의실)의 수신 상태(모델 없음, 데이터 끊김 감시만) |

`<space>` = `classroom_a`(강의실 A), `classroom_b`(강의실 B), `meeting`(회의실), `lounge`(라운지)

```bash
mosquitto_sub -h <JETSON_HOST> -p 1885 -t 'team2/svc/#' -v
```

## 2. 공간별 결과 (`team2/svc/state/<space>`, `GET /spaces/<space>`)

| 필드 | 형식 | 설명 |
|---|---|---|
| `space`, `name` | 문자열 | 공간 코드, 한글 이름 |
| `time` | 시각 | 결과 기준 시각 = 마지막으로 **끝난** 5분 칸(진행 중인 칸은 쓰지 않음) |
| `computed_at` | 시각 | 계산한 시각 |
| `now.co2` / `temperature` / `humidity` | 수 | 기준 칸의 5분 평균 (정제 후) |
| `now.filled_cells_2h` | 정수 | 최근 2시간 중 직전 값으로 채운 칸 수(짧은 결측) |
| `now.missing_cells_2h` | 정수 | 최근 2시간 중 비어 있는 칸 수(긴 결측) |
| `forecast.available` | 참/거짓 | 거짓이면 `reason`에 이유 |
| `forecast.minutes_ahead` | 정수 12개 | `[5, 10, …, 60]` |
| `forecast.co2` / `temperature` | 수 12개 | 위 시점들의 예측값 |
| `forecast.exceed_1000_in_min` | 정수/null | **몇 분 뒤 1,000ppm을 처음 넘는지**. 60분 안에 안 넘거나 이미 넘었으면 null |
| `forecast.already_over_1000` | 참/거짓 | 지금 이미 1,000ppm 이상 |
| `anomaly.score` | 수/null | 오토인코더 이상 점수 (1 ≈ 평소의 최상단). 최근 2시간에 결측 칸이 있으면 null |
| `anomaly.threshold` | 수 | 경보 기준 (1.5) |
| `anomaly.is_anomaly` | 참/거짓 | `score > threshold` |
| `anomaly.cause` | 목록 | 점수에 가장 크게 기여한 항목 상위 2개 `{"var", "z"}` |
| `anomaly.rules.<규칙>.active` | 참/거짓 | 규칙 경보 (3절) |
| `anomaly.any_alert` | 참/거짓 | 모델·규칙 중 하나라도 켜짐 |
| `state.occupied`, `occupied_prob` | 참/거짓, 0~1 | 센서만으로 추정한 사용 여부 |
| `state.vent_alert` | 참/거짓 | **환기 선제 경보**: 지금 1,000 미만이고 **30분 안에** 넘을 것으로 예측됨 |
| `state.vent_prob_classifier`, `vent_alert_classifier` | 0~1, 참/거짓 | 별도 분류기 결과(참고용, 예비) |
| `alerts_active` | 목록 | 지금 켜져 있는 경보 이름 |

## 3. 경보 이름

| 이름 | 뜻 | 기준 |
|---|---|---|
| `vent_soon` | 30분 안에 CO₂ 1,000ppm 초과 예상 | 예측 모델 |
| `co2_over_1000` | 지금 1,000ppm 이상 | 5분 평균 |
| `data_gap` | 데이터 끊김 | 마지막 수신이 10분 넘게 지남(라운지 15분). `likely_cause`: 수집기 정지 / 센서·게이트웨이 무수신 |
| `stuck` | 값 고정 | CO₂·온도가 같은 값으로 30분 이상 |
| `spike` | 튄 값 | 한 번 크게 튀었다 복귀 (최근 10분) |
| `out_of_range` | 범위 밖 값 | CO₂ 300~5,000, 온도 5~45, 습도 1~100 밖 (최근 10분 개수) |
| `co2_level` | 평소 최고치 초과 (환기 정지 의심) | 공간별 학습 기간 99.9% × 1.1 |
| `night_baseline` | 새벽 CO₂가 높음 (영점 틀어짐 의심) | 00~05시, 30분 넘게 기준 위 |
| `anomaly_model` | 오토인코더 이상 (냉방 정지 등) | 점수 > 1.5 |

### 경보 이벤트 (`team2/svc/events`, `GET /events`)

같은 경보를 매분 보내지 않는다. **처음 켜질 때 `raised`, 계속 켜져 있으면 30분마다 `reminder`, 꺼지면 `cleared`**를 한 번씩 보낸다.

```json
{"key": "classroom_a", "type": "vent_soon", "kind": "raised", "at": "2026-10-21T09:21:15", "since": "2026-10-21T09:21:15",
 "detail": {"exceed_1000_in_min": 30, "co2": 731.0}}
```
`key`는 공간 코드, 실데이터는 `real:<location>`이다.

## 4. HTTP API

| 요청 | 응답 |
|---|---|
| `GET /health` | 서비스 상태: `cycles`, `failures`, `last_cycle`, `last_elapsed_s`, `last_error`, `backends` |
| `GET /spaces` | 공간 4곳 요약: `co2_now`, `co2_60`, `exceed_1000_in_min`, `occupied`, `vent_alert`, `anomaly`, `alerts_active` |
| `GET /spaces/{space}` | 2절 전체 결과 |
| `GET /spaces/{space}/forecast?minutes=30&var=co2` | 한 시점 예측값 (`minutes` 5~60, 5분 단위 / `var` = `co2` 또는 `temperature`) |
| `GET /alerts` | 공간별 켜져 있는 경보 |
| `GET /events?limit=50` | 최근 경보 이벤트 |
| `GET /real` | 실데이터 공간·수집기 상태 |

예: "회의실 30분 뒤 CO₂는?"
```bash
curl "http://<JETSON_HOST>:8091/spaces/meeting/forecast?minutes=30"
```
```json
{"space": "meeting", "name": "회의실", "base_time": "2026-10-21T09:45:00", "minutes_ahead": 30, "var": "co2", "value": 670.4, "unit": "ppm"}
```

오류: 없는 공간은 **404**, 예측할 수 없을 때(긴 결측)는 **503**과 이유.

## 5. 실제 출력 예 (`team2/svc/state/classroom_a`, 2026-10-21 09:23:15)

```json
{"space": "classroom_a", "name": "강의실 A", "time": "2026-10-21T09:20:00", "computed_at": "2026-10-21T09:23:15",
 "now": {"co2": 731.0, "temperature": 23.74, "humidity": 64.0, "filled_cells_2h": 0, "missing_cells_2h": 0},
 "forecast": {"available": true, "minutes_ahead": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
   "co2": [802.7, 866.2, 918.1, 957.8, 996.1, 1017.8, 1003.8, 968.9, 945.3, 962.0, 997.4, 1024.9],
   "temperature": [23.81, 23.89, 23.96, 23.98, 24.01, 24.02, 23.93, 23.78, 23.78, 23.91, 24.04, 24.08],
   "already_over_1000": false, "exceed_1000_in_min": 30},
 "anomaly": {"available": true, "threshold": 1.5, "score": 0.171, "is_anomaly": false,
   "cause": [{"var": "humidity", "z": 0.171}, {"var": "temperature", "z": 0.115}],
   "rules": {"data_gap": {"active": false, "last_seen": "2026-10-21T09:23:00", "age_min": 0.2, "threshold_min": 10},
             "out_of_range": {"active": false, "count_10min": {"co2": 0, "temperature": 0, "humidity": 0}},
             "stuck": {"active": false, "minutes": null}, "spike": {"active": false, "vars": []},
             "co2_level": {"active": false, "threshold_ppm": 1412}, "night_baseline": {"active": false, "threshold_ppm": 588}},
   "any_alert": false},
 "state": {"available": true, "occupied": true, "occupied_prob": 0.998, "vent_alert": true,
   "vent_alert_source": "forecast(과제 5 결론)", "vent_prob_classifier": 0.982, "vent_alert_classifier": true},
 "alerts_active": ["vent_soon"]}
```

## 6. 로컬 LLM 에이전트가 '도구'로 쓸 때

- **질문 → 엔드포인트**: "지금 환기해야 하는 곳?" → `GET /spaces`의 `vent_alert`, "회의실 30분 뒤?" → `/spaces/meeting/forecast?minutes=30`, "센서 이상 있어?" → `GET /alerts`.
- 숫자에는 항상 **기준 시각(`time`)과 단위**가 있다. 답할 때 "09:20 기준, 30분 뒤 1,018ppm 예상"처럼 같이 말하면 된다.
- 예측을 못 하거나(`available: false`) 데이터가 끊긴 경우(`data_gap`)는 **이유가 문자열로** 들어 있어, 에이전트가 그대로 설명할 수 있다.
- 한계: 예측 구간(불확실성)은 아직 없다. 필요하면 Chronos 보조 모델(과제 6)을 붙인다.

## 7. 키트 직접 쓰기 (같은 보드의 다른 서비스)

```python
import sys; sys.path.insert(0, "/srv/team/service_kit")
import kit as K
kit = K.Kit(prefer="onnxruntime")          # 또는 "tensorrt"
results = kit.run(raw, schedule, now)      # raw: time(KST), space, co2, temperature, humidity — 최근 4시간
```
학습과 같은 전처리인지 확인: `/srv/team/venv/bin/python /srv/team/service_kit/test_consistency.py` (입력 차이 0이어야 함)
