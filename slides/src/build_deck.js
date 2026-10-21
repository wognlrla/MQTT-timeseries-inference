const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.SKILL + "/scripts/apply_theme.js");

const THEME = {
  name: "Classroom Air",
  headFontFace: "Malgun Gothic",
  bodyFontFace: "Malgun Gothic",
  colors: {
    dk1: "1B2A33", lt1: "FFFFFF", dk2: "0F2A3D", lt2: "EAF3F3",
    accent1: "2A9D8F", accent2: "F2A541", accent3: "E76F51", accent4: "6C8EAD",
    accent5: "264653", accent6: "A8DADC", hlink: "2A9D8F", folHlink: "6C8EAD",
  },
};
const HEX = THEME.colors;
const OUT = process.argv[2];

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.title = "교육장 센서 시계열 추론 — team2";
pres.author = "team2";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.0, w: 11.7, h: 1.7, fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.9, w: 11.7, h: 0.9, fontSize: 20, color: C.accent6, align: "left" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.5, 0.6, 0.5],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.9, fontSize: 30, bold: true, color: C.text2, valign: "middle" }, text: "" } },
    { text: { text: "team2 · 시계열 데이터 추론 실습", options: { x: 0.6, y: 7.0, w: 6, h: 0.3, fontSize: 10, color: C.accent4 } } },
  ],
  slideNumber: { x: 12.3, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent4 },
});
pres.defineSlideMaster({
  title: "CLOSING_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.4, w: 12.1, h: 0.9, fontSize: 32, bold: true, color: C.background1, valign: "middle" }, text: "" } },
  ],
  slideNumber: { x: 12.3, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent6 },
});

// ---------- 공통 조각 ----------
function badge(slide, x, y, label, fill, name) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: 0.5, h: 0.5, fill: { color: fill }, line: { color: fill }, objectName: name + "_badge" });
  slide.addText(label, { x, y, w: 0.5, h: 0.5, fontSize: 14, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: name + "_badge_txt" });
}
function card(slide, x, y, w, h, name, fill) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill || C.background2 }, line: { color: fill || C.background2 }, objectName: name });
}
function stat(slide, x, y, w, big, small, color, name) {
  slide.addText(big, { x, y, w, h: 0.95, fontSize: 40, bold: true, color: color || C.accent1, margin: 0, valign: "bottom", isTextBox: true, objectName: name + "_num" });
  slide.addText(small, { x, y: y + 0.98, w, h: 0.75, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: name + "_lbl" });
}
const chartBase = () => ({
  catAxisLabelColor: HEX.dk1, valAxisLabelColor: HEX.accent4, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, dataLabelFontFace: "+mn-lt", dataLabelFontSize: 12, dataLabelColor: HEX.dk1,
  valGridLine: { color: "D9E4E6", size: 0.5 }, catGridLine: { style: "none" }, showValue: true, titleFontFace: "+mn-lt",
  titleColor: HEX.dk2, titleFontSize: 13, showTitle: true,
});

// ---------- 1. 제목 ----------
pres.addSection({ title: "도입" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "도입" });
s.addText("교육장 센서로 ‘지금’과 ‘60분 뒤’를 읽는다", { placeholder: "title" });
s.addText("MQTT 시계열 추론 · Jetson 상시 서비스  |  team2 기본 트랙", { placeholder: "body" });
[["수집", "유실 0"], ["예측·탐지·추정", "확인 기준 모두 통과"], ["상시 서비스", "매분 0.4초"]].forEach((t, i) => {
  const x = 0.8 + i * 3.95;
  badge(s, x, 5.45, String(i + 1), C.accent2, "t" + i);
  s.addText([{ text: t[0], options: { bold: true, color: C.background1, breakLine: true } }, { text: t[1], options: { color: C.accent6 } }],
    { x: x + 0.65, y: 5.3, w: 3.1, h: 0.85, fontSize: 15, margin: 0, valign: "middle", isTextBox: true, objectName: "title_step" + i });
});
s.addNotes("교육장 CO2·온도·습도 센서만으로 지금 상태와 60분 뒤를 판단하는 서비스를 Jetson 한 대에서 만들었습니다. 수집부터 예측, 이상 탐지, 상태 추정, 엣지 최적화, 상시 서비스까지 과제 1~8을 기본 트랙으로 완료했습니다.");

// ---------- 2. 구성 ----------
pres.addSection({ title: "구성" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "구성" });
s.addText("센서에서 판단까지, 보드 한 대 안에서", { placeholder: "title" });
const flow = [["센서", "LoRaWAN\nCO₂·온도·습도"], ["MQTT 수집기", "설정 파일로\n형식 대응"], ["InfluxDB", "원본 그대로\n저장"], ["추론 키트", "공통 전처리\nGRU·AE·규칙"], ["MQTT · API", "retain 발행\n:8091 조회"]];
flow.forEach((f, i) => {
  const x = 0.6 + i * 2.5;
  card(s, x, 1.65, 2.05, 1.75, "flow_card" + i, i === 3 ? C.accent1 : C.background2);
  s.addText([{ text: f[0], options: { bold: true, fontSize: 16, breakLine: true } }, { text: f[1], options: { fontSize: 12 } }],
    { x: x + 0.12, y: 1.75, w: 1.81, h: 1.55, color: i === 3 ? C.background1 : C.text1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "flow_txt" + i });
  if (i < flow.length - 1) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 2.1, y: 2.35, w: 0.35, h: 0.35, fill: { color: C.accent4 }, line: { color: C.accent4 }, objectName: "flow_arrow" + i });
});
[["0.4초", "매분 4개 공간 예측·탐지·추정 한 주기"], ["0", "학습과 서비스의 모델 입력 차이\n(159개 시점 비교)"], ["0건", "수집기 정지·재부팅·Docker 교체 중\n잃어버린 데이터"]].forEach((t, i) => {
  const x = 0.6 + i * 4.15;
  card(s, x, 3.95, 3.85, 2.55, "stat_card" + i);
  stat(s, x + 0.35, 4.15, 3.2, t[0], t[1], i === 2 ? C.accent2 : C.accent1, "s2_" + i);
});
s.addNotes("구성은 한 줄입니다. 센서 메시지를 MQTT 수집기가 받아 InfluxDB에 원본 그대로 저장하고, 추론 키트가 학습 때와 똑같은 전처리를 거쳐 GRU 예측, 오토인코더와 규칙 탐지, 상태 분류를 돌립니다. 결과는 MQTT retain과 조회 API로 나갑니다. 한 주기 0.4초, 학습-서비스 입력 차이 0, 데이터 유실 0이 핵심 숫자입니다.");

// ---------- 3. 수집 ----------
pres.addSection({ title: "결과" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "결과" });
s.addText("수집: 끊겨도 데이터는 잃지 않는다", { placeholder: "title" });
[["재부팅 2회", "부팅 21~23초 뒤 자동 재연결, 공백 약 45초"], ["브로커 재시작", "다시 켜진 뒤 0.6초 만에 재연결"], ["Docker 교체 사고", "JetPack 설치가 Docker를 바꿔 5분 정지 → 브로커 보관분 재전송, 유실 0"], ["수집기 10분 정지", "재시작 뒤 10분치 모두 채워짐"]].forEach((t, i) => {
  const y = 1.55 + i * 1.3;
  badge(s, 0.6, y + 0.1, String(i + 1), C.accent1, "c" + i);
  s.addText([{ text: t[0], options: { bold: true, fontSize: 16, breakLine: true } }, { text: t[1], options: { fontSize: 14 } }],
    { x: 1.3, y, w: 5.2, h: 1.1, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: "event" + i });
});
s.addChart(pres.charts.BAR, [{ name: "1분 커버율(%)", labels: ["실습실", "사무실", "사무실 밖", "회의실"], values: [100, 99.4, 77.8, 9.6] }], Object.assign(chartBase(), {
  x: 6.9, y: 1.45, w: 5.9, h: 4.4, barDir: "bar", chartColors: [HEX.accent1], showLegend: false, valAxisMaxVal: 100, valAxisMinVal: 0,
  title: "24시간 온도 1분 커버율 (%) — 10/20 10:21~10/21 10:21", dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", catAxisOrientation: "maxMin", objectName: "coverage_chart",
}));
s.addText("회의실은 10분마다, 사무실 밖은 1대가 약 1분마다 보내서 낮게 나온다. 센서별 수신률은 96.5~104%.", {
  x: 6.9, y: 5.95, w: 5.9, h: 0.8, fontSize: 12, color: C.accent4, margin: 0, isTextBox: true, objectName: "coverage_note" });
s.addNotes("수집기는 고정 클라이언트 ID와 영속 세션, QoS 1로 접속해서 끊긴 동안의 메시지를 브로커가 보관합니다. 재부팅, 브로커 재시작, JetPack 설치가 Docker를 바꿔 버린 사고, 일부러 10분 멈춘 시험까지 모두 데이터 유실이 없었습니다. 커버율이 낮은 두 공간은 수집 문제가 아니라 센서 보고 주기 때문입니다.");

// ---------- 4. 예측 ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "결과" });
s.addText("예측: 일정을 알면 회의실이 보인다", { placeholder: "title" });
s.addChart(pres.charts.BAR, [{ name: "60분 뒤 CO₂ MAE", labels: ["최근 1시간 평균", "현재값 유지", "GRU 일정 제외", "GRU 일정 포함"], values: [54.0, 45.3, 24.3, 18.0] }], Object.assign(chartBase(), {
  x: 0.6, y: 1.45, w: 7.0, h: 5.2, barDir: "col", chartColors: [HEX.accent4, HEX.accent4, HEX.accent6, HEX.accent1], showLegend: false,
  title: "60분 뒤 CO₂ 오차 (ppm, 시험 기간·고장 제외)", dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMinVal: 0, objectName: "forecast_chart",
}));
card(s, 8.0, 1.45, 4.75, 2.4, "fc_card1");
stat(s, 8.35, 1.6, 4.1, "−60%", "현재값 유지 대비 (45.3 → 18.0ppm)\n기준: 30% 이상 개선", C.accent1, "s4a");
card(s, 8.0, 4.15, 4.75, 2.5, "fc_card2");
stat(s, 8.35, 4.3, 4.1, "회의실 −40%", "일정 없이는 기준선과 비슷(26.3 vs 27.4)\n회의 시작은 센서가 아니라 예약에 있다", C.accent2, "s4b");
s.addNotes("과거 2시간과 앞으로 60분의 일정을 넣은 GRU가 60분 뒤 CO2 오차를 45에서 18ppm으로 60% 줄였습니다. 일정 정보의 가치가 가장 큰 곳은 회의실입니다. 일정 없이는 기준선과 거의 같았는데, 일정을 넣자 40% 좋아졌습니다. 회의가 언제 시작하는지는 센서값이 아니라 예약 시스템에만 있기 때문입니다.");

// ---------- 5. 이상 탐지 ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "결과" });
s.addText("이상 탐지: 모델과 규칙이 서로의 빈칸을 메운다", { placeholder: "title" });
const hit = (v) => ({ text: v, options: { align: "center", color: v.startsWith("0") ? C.accent3 : C.text1, bold: !v.startsWith("0") } });
const hdr = (v) => ({ text: v, options: { bold: true, color: C.background1, fill: { color: C.text2 }, align: "center" } });
s.addTable([
  [hdr("고장 (시험 17건)"), hdr("오토인코더"), hdr("규칙"), hdr("합침")],
  [{ text: "튄 값", options: { bold: true } }, hit("1/6"), hit("5/6"), hit("5/6")],
  [{ text: "값 고정", options: { bold: true } }, hit("1/3"), hit("2/3"), hit("2/3")],
  [{ text: "영점 틀어짐", options: { bold: true } }, hit("1/2"), hit("2/2"), hit("2/2")],
  [{ text: "냉방 정지", options: { bold: true } }, hit("3/3"), hit("0/3"), hit("3/3")],
  [{ text: "환기 정지", options: { bold: true } }, hit("0/3"), hit("2/3"), hit("2/3")],
], { x: 0.6, y: 1.5, w: 7.2, colW: [2.4, 1.6, 1.6, 1.6], rowH: 0.62, fontSize: 15, color: C.text1, fill: { color: C.background2 },
  border: { type: "solid", pt: 1, color: HEX.lt1 }, valign: "middle", objectName: "detect_table" });
s.addText("놓친 3건 중 2건은 데이터에 없었다 — 5분 기록 사이의 튄 값, 게이트웨이 끊김과 겹친 값 고정", {
  x: 0.6, y: 5.45, w: 7.2, h: 0.9, fontSize: 14, color: C.accent4, margin: 0, isTextBox: true, objectName: "missed_note" });
card(s, 8.25, 1.5, 4.5, 2.35, "det_card1");
stat(s, 8.6, 1.65, 3.9, "82%", "합친 이벤트 탐지율 (14/17)\n기준 80%", C.accent1, "s5a");
card(s, 8.25, 4.1, 4.5, 2.35, "det_card2");
stat(s, 8.6, 4.25, 3.9, "0.29회/일", "헛경보 (4개 공간 합)\n기준 하루 1회 이하", C.accent1, "s5b");
s.addNotes("정상만 배운 조건부 오토인코더는 냉방 정지 3건을 모두 잡았지만 튄 값과 값 고정에는 약했습니다. 반대로 규칙은 튄 값과 값 고정을 헛경보 없이 잡았지만 냉방 정지는 하나도 못 잡았습니다. 둘을 합쳐 82%, 헛경보 하루 0.29회로 기준을 넘었습니다. 놓친 세 건 중 두 건은 애초에 데이터에 남지 않은 고장이었습니다.");

// ---------- 6. 상태 추정 ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "결과" });
s.addText("상태 추정: 틀린 정답으로 배운 모델이 정답보다 정확했다", { placeholder: "title" });
s.addChart(pres.charts.BAR, [{ name: "F1 (사용중)", labels: ["규칙 15분 +30ppm", "일정 그대로", "모델 (센서만)"], values: [0.657, 0.752, 0.842] }], Object.assign(chartBase(), {
  x: 0.6, y: 1.45, w: 6.6, h: 4.4, barDir: "col", chartColors: [HEX.accent4, HEX.accent6, HEX.accent1], showLegend: false,
  title: "사용 여부 F1 — 직접 관찰 814칸 기준", dataLabelPosition: "outEnd", valAxisMinVal: 0, valAxisMaxVal: 1, dataLabelFormatCode: "0.000", objectName: "occ_chart",
}));
s.addText("정답을 만든 일정 자체가 11.4% 틀렸다. 일정을 입력에서 빼자 모델은 일정의 실수 대신 CO₂의 물리적 흔적을 배웠다.", {
  x: 0.6, y: 5.95, w: 6.6, h: 0.8, fontSize: 13, color: C.accent4, margin: 0, isTextBox: true, objectName: "occ_note" });
card(s, 7.6, 1.45, 5.15, 2.4, "st_card1");
stat(s, 7.95, 1.6, 4.5, "0.2%", "빈 방을 ‘사용중’으로 잘못 본 비율\n(냉난방 자동 제어에 가장 중요)", C.accent1, "s6a");
card(s, 7.6, 4.15, 5.15, 2.6, "st_card2");
stat(s, 7.95, 4.3, 4.5, "재현율 0.86", "환기 선제 경보 (기준 0.85)\n예측 모델로 판단하면 0.90 → 서비스에 채택", C.accent2, "s6b");
s.addNotes("사용 여부의 정답은 수업 시간표와 예약으로 만들었는데, 직접 관찰과 비교하니 11% 이상 틀렸습니다. 그런데 일정을 입력에서 빼고 센서 변화만 본 모델은 일정 자체보다 실제 사용을 더 잘 맞혔습니다. 빈 방 오판은 0.2%입니다. 환기 선제 경보는 분류기도 기준을 넘었지만, 과제 3 예측 모델로 판단하는 쪽이 더 정확해서 서비스에는 예측 모델을 썼습니다.");

// ---------- 7. 엣지 ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "결과" });
s.addText("엣지: 작은 모델은 GPU보다 CPU가 빠르다", { placeholder: "title" });
s.addChart(pres.charts.BAR, [
  { name: "PyTorch GPU", labels: ["예측 GRU", "예측 TCN", "오토인코더"], values: [1.10, 2.14, 1.13] },
  { name: "TensorRT FP16", labels: ["예측 GRU", "예측 TCN", "오토인코더"], values: [1.27, 0.62, 0.47] },
  { name: "ONNX Runtime CPU", labels: ["예측 GRU", "예측 TCN", "오토인코더"], values: [0.21, 0.12, 0.05] },
], Object.assign(chartBase(), {
  x: 0.6, y: 1.45, w: 7.2, h: 5.2, barDir: "col", barGrouping: "clustered", chartColors: [HEX.accent4, HEX.accent2, HEX.accent1],
  showLegend: true, legendPos: "b", legendFontFace: "+mn-lt", legendFontSize: 12, legendColor: HEX.dk1,
  title: "1회 추론 지연 (ms, 배치 1, 25W)", dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", valAxisMinVal: 0, objectName: "latency_chart",
}));
[["GPU로 보내고 받는 고정 비용(~0.35ms)이 계산보다 크다", "그래서 서비스는 ONNX Runtime CPU로 운영 — 메모리도 0.5GB 절약"],
 ["TCN은 TensorRT로 3.5배, GRU는 그대로", "합성곱은 병렬, GRU는 24칸을 차례로 계산"],
 ["FP16 오차 1.31ppm → 출력부만 FP32로 0.71ppm", "기준 1ppm 통과, 분류 결과 변화 0%"]].forEach((t, i) => {
  const y = 1.45 + i * 1.75;
  badge(s, 8.2, y + 0.05, String(i + 1), i === 2 ? C.accent2 : C.accent1, "e" + i);
  s.addText([{ text: t[0], options: { bold: true, fontSize: 15, breakLine: true } }, { text: t[1], options: { fontSize: 13, color: C.accent4 } }],
    { x: 8.85, y, w: 3.95, h: 1.55, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: "edge_pt" + i });
});
s.addNotes("TensorRT로 바꾸면 빨라질 거라 예상했지만, 이 크기의 모델은 ONNX Runtime CPU가 5~14배 더 빨랐습니다. GPU로 데이터를 보내고 받는 고정 비용이 계산보다 크기 때문입니다. TCN은 TensorRT로 3.5배 빨라졌지만 GRU는 순차 계산이라 그대로였습니다. FP16 오차는 출력부만 FP32로 두어 0.71ppm으로 기준을 통과했습니다.");

// ---------- 8. 서비스 ----------
pres.addSection({ title: "서비스" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "서비스" });
s.addText("상시 서비스 v1: 매분, 끊겨도, 다시 켜져도", { placeholder: "title" });
[["매분 계산", "60분 예측 · ‘몇 분 뒤 1,000ppm 초과’ · 이상 점수·원인 · 사용 여부"],
 ["두 가지로 제공", "MQTT retain(새 구독자 즉시 13개 토픽) + 조회 API :8091"],
 ["학습과 같은 전처리", "같은 함수를 import — 모델 입력 차이 0"],
 ["장애 시험", "계산 실패 → 다음 주기 정상 / 수집기 10분 정지 → data_gap 경보 → 재시작 49초 뒤 해제"],
 ["자동 시작·12시간", "systemd + Docker 자동 재시작, 10/21 09:45 가동 시작"]].forEach((t, i) => {
  const y = 1.45 + i * 1.08;
  s.addShape(pres.shapes.OVAL, { x: 0.6, y: y + 0.08, w: 0.42, h: 0.42, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "chk" + i });
  s.addText("✓", { x: 0.6, y: y + 0.08, w: 0.42, h: 0.42, fontSize: 14, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "chk_t" + i });
  s.addText([{ text: t[0], options: { bold: true, fontSize: 15, breakLine: true } }, { text: t[1], options: { fontSize: 13 } }],
    { x: 1.2, y, w: 6.2, h: 0.98, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: "svc_pt" + i });
});
card(s, 7.8, 1.45, 4.95, 5.2, "demo_card", C.text2);
s.addText("GET /spaces", { x: 8.1, y: 1.65, w: 4.4, h: 0.45, fontSize: 15, bold: true, color: C.accent2, margin: 0, isTextBox: true, objectName: "demo_req" });
s.addText([
  { text: "강의실 A  09:20 기준", options: { bold: true, breakLine: true } },
  { text: "지금 CO₂  731ppm", options: { breakLine: true } },
  { text: "30분 뒤  1,018ppm 예상", options: { breakLine: true } },
  { text: "exceed_1000_in_min: 30", options: { color: C.accent2, breakLine: true } },
  { text: "occupied: true (0.998)", options: { breakLine: true } },
  { text: "alerts: vent_soon", options: { color: C.accent2 } },
], { x: 8.1, y: 2.25, w: 4.4, h: 3.0, fontSize: 15, color: C.background1, margin: 0, valign: "top", paraSpaceAfter: 6, isTextBox: true, objectName: "demo_resp" });
s.addText("수업 시작 20분 뒤, ‘30분 안에 1,000ppm’을 미리 알렸다 (실제 서비스 출력)", {
  x: 8.1, y: 5.45, w: 4.4, h: 1.0, fontSize: 12, color: C.accent6, margin: 0, isTextBox: true, objectName: "demo_note" });
s.addNotes("상시 서비스는 매분 15초에 4개 공간을 계산해 MQTT와 API로 내보냅니다. 학습 때 쓴 함수를 그대로 불러와 입력 차이가 0인 것을 시험으로 확인했고, 계산을 일부러 실패시켜도, 수집기를 10분 멈춰도 다음 주기에 정상으로 돌아왔습니다. 오른쪽은 실제 출력입니다. 수업이 시작되고 20분 뒤, 30분 안에 1,000ppm을 넘는다고 미리 알렸습니다.");

// ---------- 9. 의외의 발견 ----------
pres.addSection({ title: "마무리" });
s = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: "마무리" });
s.addText("가장 의외였던 발견", { placeholder: "title" });
[["틀린 정답이 더 나은 모델을 만들었다", "11% 틀린 일정 정답으로 배운 센서 모델이 일정보다 실제 사용을 더 잘 맞혔다 (F1 0.84 vs 0.75)"],
 ["냉방 고장은 ‘습도’로 먼저 보였다", "냉방이 멈추면 제습도 멈춘다 — 냉방 정지 3건 중 2건에서 오토인코더가 습도를 먼저 가리켰다"],
 ["가장 위험한 건 모델이 아니라 메모리", "사전학습 모델 하나가 보드를 5분 멈췄고, 같은 보드의 작업들 때문에 서비스 실행 방식을 바꿨다"]].forEach((t, i) => {
  const x = 0.6 + i * 4.15;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.6, w: 3.85, h: 3.45, rectRadius: 0.12, fill: { color: C.accent5 }, line: { color: C.accent5 }, objectName: "find_card" + i });
  badge(s, x + 0.35, 1.9, String(i + 1), C.accent2, "f" + i);
  s.addText(t[0], { x: x + 0.35, y: 2.6, w: 3.15, h: 1.1, fontSize: 19, bold: true, color: C.background1, margin: 0, valign: "top", isTextBox: true, objectName: "find_h" + i });
  s.addText(t[1], { x: x + 0.35, y: 3.7, w: 3.15, h: 1.25, fontSize: 15, color: C.accent6, margin: 0, valign: "top", isTextBox: true, objectName: "find_b" + i });
});
s.addText("다음: 실데이터(과거 15개월 이관 완료)로 재학습 · 스왑 추가 · 심화 과제", {
  x: 0.6, y: 5.45, w: 12.1, h: 0.5, fontSize: 16, color: C.accent6, margin: 0, isTextBox: true, objectName: "next_steps" });
s.addNotes("가장 의외였던 세 가지입니다. 첫째, 11% 틀린 정답으로 배운 모델이 정답보다 정확했습니다. 둘째, 냉방 고장을 모델은 온도보다 습도로 먼저 알아챘습니다. 셋째, 실제로 가장 위험했던 것은 모델 정확도가 아니라 보드 메모리였습니다. 다음 단계는 이미 옮겨 둔 실데이터 15개월로 재학습하는 것입니다. 감사합니다.");

pres.writeFile({ fileName: OUT }).then(async () => {
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
});
