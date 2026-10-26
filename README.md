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
| Yeniden sıralama | bge-reranker-v2-m3 |
| Vektör veritabanı | ChromaDB |
| Arama | Yoğun (bge-m3) + BM25, çapraz kodlayıcı ile yeniden sıralama, birim/bölüm öncelikleri |
| Arayüz | FastAPI + tek sayfa HTML (`web/index.html`) |

## Veri

- KOÜ web siteleri (`*.kocaeli.edu.tr`): sayfalar, duyurular, PDF/DOCX dosyaları, AKTS program ve ders bilgileri
- mevzuat.gov.tr: ilgili kanun ve yönetmelikler
- Toplam ~8.500 belge, ~72.000 madde/bölüm, 39 birim

Kişisel listeler (sınav/yerleşim sonuçları, personel ad-telefon listeleri) ve KOÜ ile ilgisiz içerik veri hattında
elenir. Veri ve veritabanı boyut nedeniyle depoda yoktur; aşağıdaki betiklerle yeniden oluşturulur.

## Klasör yapısı

Depo `proje/` klasörüdür; betikler veriyi bir üst klasördeki `veri/` altında, veritabanını `~/kou_rag_chroma`
altında bekler.

```
KOÜ_RAG/
├── proje/        bu depo
├── veri/         belgeler, işlenmiş maddeler, eğitim ve test verisi
└── araclar/      llama.cpp (GGUF dönüştürme ve nicemleme)
```

## Çalıştırma

```bash
pip install -r requirements.txt
ollama pull bge-m3
python -X utf8 web_app.py        # http://127.0.0.1:8000
```

## Veri hattı

| Adım | Betik |
|---|---|
| Web taraması | `web_tara_tam.py`, `web_dosya_indir.py`, `dosya_metin.py`, `tablo_metin.py` |
| Mevzuat ve AKTS | `mevzuat_tara.py`, `akts_tara.py`, `dersler_isle.py` |
| Katalog (eleme, birim ataması) | `tam_katalog.py`, `birim_haritasi.py`, `web_yapi.py` |
| PDF belgeleri | `veri_duzenle.py`, `degisiklikleri_cikar.py`, `konu_etiketle.py` |
| Madde/parça + veritabanı | `belge_isle.py` (`--artimli`: yalnız değişen parçalar) |
| İlgisiz içerik temizliği | `ilgisiz_bul.py`, `ilgisiz_sil.py` |

## Arama ve sohbet

- `rag_v3.py`: arama hattı ve sohbet (birim/bölüm netleştirme soruları, karar ağaçları, harç ücret tablosu,
  devam soruları)
- `kaynak_dogrula.py`: kaynak satırını doğrular/düzeltir
- `sayi_kontrol.py`: cevaptaki sayılar kaynakta yoksa uyarı ekler
- `belge_bilgisi.py`: belgenin orijinal bağlantısı ve tarihi

## Eğitim (v4)

```bash
python -X utf8 generate_raft_data_v4.py
.venv-train/Scripts/python.exe train_qlora.py --v2 --v4
.venv-train/Scripts/python.exe export_ollama.py --v2 --v4
```

`gece_v4.sh` tüm adımları sırayla çalıştırır.

## Değerlendirme

```bash
bash degerlendir_hepsi.sh <etiket> --sadece --model kou-asistan-v4
```

Üç test seti (genel mevzuat, web belgeleri, yeni konular; 174 cevaplanabilir soru + kapsam dışı sorular), hakem
modeliyle puanlanır. Son ölçüm: **152/174 (%87,4)**, kapsam dışı soruları reddetme 7/8.
