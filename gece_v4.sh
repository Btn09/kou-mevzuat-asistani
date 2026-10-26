#!/usr/bin/env bash
# v4 gece hattı: veri üretimi -> (web sunucusu durdurulur) eğitim -> Ollama'ya aktarma -> testler.
# Çalıştırma: bash gece_v4.sh   (Git Bash; proje klasöründe). Durum: ../veri/egitim_v4/gece_durum.txt
set -u
cd "$(dirname "$0")"
OUT=../veri/egitim_v4
LOG=../veri/test_v3
mkdir -p $OUT
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONUNBUFFERED=1
say() { echo "[$(date +%H:%M)] $*" >> $OUT/gece_durum.txt; }

say "veri üretimi başlıyor"
python -X utf8 generate_raft_data_v4.py > $OUT/uretim_log.txt 2>&1 || { say "VERİ ÜRETİM HATASI"; exit 1; }
say "veri hazır: $(wc -l < $OUT/raft_train_v4.jsonl) eğitim örneği"

# Eğitim GPU belleğinin ~%85'ini kullanır: web sunucusu (reranker + sohbet modeli) ve Ollama modelleri boşaltılır.
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*web_app.py*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
for m in qwen3:14b kou-asistan-v2 bge-m3 trendyol-8b-ham; do ollama stop $m > /dev/null 2>&1; done
say "web sunucusu durduruldu, eğitim başlıyor"
./.venv-train/Scripts/python.exe -X utf8 train_qlora.py --v2 --v4 > $OUT/egitim_log.txt 2>&1 \
  || { say "EĞİTİM HATASI"; exit 1; }
say "eğitim bitti, aktarma"
./.venv-train/Scripts/python.exe -X utf8 export_ollama.py --v2 --v4 > $OUT/export_log.txt 2>&1 \
  || { say "AKTARMA HATASI"; exit 1; }
rm -rf trendyol-8b-kou-merged-v4 kou-asistan-v4-bf16.gguf
say "aktarma bitti, test (eski test seti)"
python -X utf8 degerlendir_v3.py v4_model --hat v3 --model kou-asistan-v4 > $LOG/log_v4_model.txt 2>&1
say "test (web test seti)"
python -X utf8 degerlendir_v3.py v4_model_web --hat v3 --set web --model kou-asistan-v4 > $LOG/log_v4_model_web.txt 2>&1
say "TAMAM"
