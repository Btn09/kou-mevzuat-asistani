# KOÜ Mevzuat Asistanı

Kocaeli Üniversitesi'nin yönetmelik, yönerge, duyuru, akademik takvim ve ders bilgileri üzerinden soru cevaplayan,
her cevabında dayandığı maddeyi gösteren bir asistan. Öğrenciler ve akademik/idari personel için tek sistem.

Yaklaşım: **RAG + RAFT** — belgeler arama ile bulunur (RAG), cevap modeli bu arama bağlamıyla ince ayar yapılmıştır
(RAFT: doğru maddeyi seçme, kaynak gösterme, cevap yoksa reddetme).

## Bileşenler

| Bileşen | Seçim |
|---|---|
| Cevap modeli | Trendyol-LLM-8B-T1 + QLoRA ince ayarı (`kou-asistan-v4`, Ollama, Q4_K_M) |
| Gömme modeli | bge-m3 (Ollama) |
| Yeniden sıralama | BAAI/bge-reranker-v2-m3 |
| Vektör veritabanı | ChromaDB |
| Arama | Yoğun (bge-m3) + BM25, çapraz kodlayıcı ile yeniden sıralama, birim/bölüm öncelikleri |
| Veri üretimi ve hakem | qwen3:14b (Ollama; yalnız eğitim verisi üretimi ve değerlendirme) |
| Arayüz | FastAPI + tek sayfa HTML (`web/index.html`) |

Son ölçüm: **152/174 (%87,4)** doğru cevap, kapsam dışı soruları reddetme 7/8.

---

## Kurulum

Depoda yalnız kod vardır. Veri (~8 GB), vektör veritabanı (~2,3 GB) ve model ağırlıkları boyut nedeniyle depoda
yoktur; aşağıdaki adımlarla üretilir.

### 1. Gereksinimler

- Windows 10/11 veya Linux, Python 3.12
- NVIDIA ekran kartı (eğitim için en az 16 GB bellek; yalnız çalıştırmak için 8 GB yeterli)
- [Ollama](https://ollama.com)
- Disk: ~60 GB (veri + modeller + eğitim ara dosyaları)

```bash
pip install -r requirements.txt
ollama pull bge-m3
ollama pull qwen3:14b
```

Eğitim ayrı bir sanal ortamda yapılır (`.venv-train`; `requirements.txt` sonundaki eğitim paketleri: peft,
bitsandbytes, accelerate, datasets). GGUF dönüştürme için [llama.cpp](https://github.com/ggml-org/llama.cpp)
gerekir: `convert_hf_to_gguf.py` ve `llama-quantize` (`export_ollama.py` bunları `../araclar/` altında arar).

### 2. Klasör düzeni

Betikler veriyi deponun bir üst klasöründeki `veri/` altında, veritabanını `~/kou_rag_chroma` altında bekler
(değiştirmek için `KOU_CHROMA_DIR` ortam değişkeni):

```
KOU_RAG/
├── proje/        bu depo (git clone ... proje)
├── veri/         betiklerin ürettiği veri
├── yeni_pdfler/  (isteğe bağlı) elle eklenen yönetmelik PDF'leri
├── new_veri/     (isteğe bağlı) fakülte/MYO belgeleri
└── araclar/      llama.cpp
```

### 3. Veriyi oluşturma

Sırayla çalıştırın (tümü `proje/` içinde, `python -X utf8 <betik>`):

| # | Betik | Ne yapar | Süre |
|---|---|---|---|
| 1 | `web_tara_tam.py` | KOÜ sitelerini tarar (sayfa, duyuru, dosya bağlantıları) → `veri/web_tam/` | ~2-3 saat |
| 2 | `web_dosya_indir.py` | Sitelerdeki PDF/DOCX dosyalarını indirir | ~1-2 saat |
| 3 | `dosya_metin.py` | Dosyalardan metin çıkarır | ~30 dk |
| 4 | `tablo_metin.py` | Ücret ve takvim tablolarını satır satır metne çevirir | ~10 dk |
| 5 | `mevzuat_tara.py` | mevzuat.gov.tr'den ilgili kanun/yönetmelikler | ~5 dk |
| 6 | `akts_tara.py` | AKTS program ve ders bilgileri | ~30 dk |
| 7 | `web_yapi.py` | Akademik yapı (birimler, bölümler) | ~5 dk |
| 8 | `birim_haritasi.py` | Bölüm/program adı → fakülte haritası | saniyeler |
| 9 | `tam_katalog.py` | Belge kataloğu: kopya, kısa, kişisel liste, KOÜ dışı içerik elenir; birim atanır | ~5 dk |
| 10 | `belge_isle.py` | Madde/parça bölme + bge-m3 gömme + ChromaDB (`--artimli`: yalnız değişen parçalar) | ~2-3 saat |
| 11 | `dersler_isle.py` | Ders bilgileri ayrı koleksiyona | ~1 saat |

İsteğe bağlı:
- `ilgisiz_bul.py` + `ilgisiz_sil.py`: KOÜ ile ilgisiz içeriği bulup çıkarır, sonra 9 ve 10 tekrarlanır.
- `veri_duzenle.py`, `degisiklikleri_cikar.py`, `konu_etiketle.py`: elle verilen PDF belgeleri (yönetmelik
  arşivi) için sınıflandırma, değişiklik işleme, konu etiketleri. Kaynak PDF'ler deponun yanındaki
  `yeni_pdfler/` ve `new_veri/` klasörlerinde beklenir (KOÜ'nün yayımladığı yönetmelik ve yönergeler).

### 4. Cevap modelini eğitme (v4)

Veri üretimi önceki sürümlerin verisini de kullanır, bu yüzden sırayla çalıştırılır:

```bash
python -X utf8 test_seti_uret.py                 # test seti (test maddeleri eğitimden hariç tutulur)
python -X utf8 generate_raft_data_v2.py --tum    # temel soru-cevap verisi     (~8 saat)
python -X utf8 generate_raft_data_v3.py          # devam sorusu örnekleri      (~3 saat)
python -X utf8 generate_raft_data_v4.py          # güncel veriyle ayrıntılı örnekler (~7 saat)
.venv-train/Scripts/python.exe train_qlora.py --v2 --v4     # QLoRA eğitimi (~5 saat)
.venv-train/Scripts/python.exe export_ollama.py --v2 --v4   # GGUF + Ollama modeli kou-asistan-v4
```

`gece_v4.sh` son üç adımı (veri üretimi, eğitim, aktarma, test) arka arkaya çalıştırır.

### 5. Çalıştırma

```bash
python -X utf8 web_app.py        # http://127.0.0.1:8000
```

- Ollama açık olmalı (`kou-asistan-v4` ve `bge-m3`).
- `KOU_ERISIM_KODU` ortam değişkeni verilirse site girişte bu kodu sorar (internete açarken kullanın).
- Aynı ağdaki cihazlardan erişim için `web_app.py` içinde `HOST = "0.0.0.0"`.

### 6. Değerlendirme

```bash
bash degerlendir_hepsi.sh <etiket> --sadece --model kou-asistan-v4
```

Üç test seti (genel mevzuat, web belgeleri, yeni konular; 174 cevaplanabilir soru + kapsam dışı sorular)
qwen3:14b hakemiyle puanlanır. Sonuçlar `veri/test_v3/` altına yazılır.

---

## Kod yapısı

| Dosya | Görev |
|---|---|
| `web_app.py`, `web/` | Web sunucusu ve arayüz |
| `rag_v3.py` | Arama hattı ve sohbet: birim/bölüm netleştirme soruları, karar ağaçları, harç ücret tablosu, devam soruları |
| `rag_v2.py` | Birim tanıma, yeniden sıralayıcı, ortak yardımcılar |
| `kaynak_dogrula.py` | Kaynak satırını doğrular/düzeltir |
| `sayi_kontrol.py` | Cevaptaki sayılar kaynakta yoksa uyarı ekler |
| `belge_bilgisi.py` | Belgenin orijinal bağlantısı ve tarihi |
| `raft_format.py` | Model istemi ve bağlam biçimi (eğitim ve çalışma zamanında aynı) |
| `generate_raft_data*.py`, `train_qlora.py`, `export_ollama.py` | Eğitim verisi, eğitim, Ollama'ya aktarma |
| `degerlendir_v3.py`, `hakem_degerlendir.py` | Test ve hakem puanlaması |
