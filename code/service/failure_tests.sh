#!/bin/sh
# 과제 8 장애 시험: ① 한 주기 계산 실패 → 다음 주기 회복 ② 수집기-sim 10분 정지 → 데이터 끊김 경보 → 재시작 후 해제
cd /srv/team/service
L=failure_tests.log
echo "$(date +%T) ① 계산 실패 시험 시작: FAIL_NEXT 생성" > $L
touch FAIL_NEXT
sleep 150
curl -s localhost:8091/health >> $L; echo >> $L
grep -E "주기 실패|RuntimeError|주기 완료" service.log | tail -4 >> $L
echo "$(date +%T) ② 수집기-sim 정지" >> $L
docker stop svc-collector-sim >> $L 2>&1
STOP=$(date +%s)
while [ $(( $(date +%s) - STOP )) -lt 600 ]; do
  sleep 60
  echo "$(date +%T) 정지 $(( ($(date +%s) - STOP) / 60 ))분: $(curl -s localhost:8091/alerts)" >> $L
done
echo "$(date +%T) 수집기-sim 재시작" >> $L
docker start svc-collector-sim >> $L 2>&1
for i in 1 2 3 4 5 6; do sleep 60; echo "$(date +%T) 재시작 후 ${i}분: $(curl -s localhost:8091/alerts)" >> $L; done
grep "경보" service.log | tail -20 >> $L
echo TESTS_DONE >> $L
