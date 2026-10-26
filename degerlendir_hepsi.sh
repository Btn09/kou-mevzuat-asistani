#!/bin/bash
# Üç test seti (eski, web, yeni) güncel hatla: bash degerlendir_hepsi.sh <etiket> [--sadece] [--model <ad>]
# --sadece yoksa önce kayıtlı b4 cevapları yeni hakemle yeniden puanlanır (karşılaştırma için, 23.10).
cd "$(dirname "$0")"
ET=${1:-c1}
shift
ONLY=0
MODEL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --sadece) ONLY=1 ;;
    --model) MODEL="--model $2"; shift ;;
  esac
  shift
done
if [ $ONLY = 0 ]; then
  for S in eski web yeni; do
    SET=""; [ $S = web ] && SET="--set web"; [ $S = yeni ] && SET="--set yeni"
    python -X utf8 degerlendir_v3.py b4h_$S --hat v3 $SET --kayitli > ../veri/test_v3/log_b4h_$S.txt 2>&1
  done
fi
for S in eski web yeni; do
  SET=""; [ $S = web ] && SET="--set web"; [ $S = yeni ] && SET="--set yeni"
  python -X utf8 degerlendir_v3.py ${ET}_$S --hat v3 $SET $MODEL > ../veri/test_v3/log_${ET}_$S.txt 2>&1
done
echo bitti > ../veri/test_v3/durum_$ET.txt
