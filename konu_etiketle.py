#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Her maddeye tek bir ana konu etiketi verir (qwen3:14b) -> veri/islenmis/konu_etiketleri.json

belge_isle.py bu dosyayı okuyup parçalardaki "konu" alanını doldurur. Etiketler madde düzeyinde:
uzun maddelerin parçaları aynı konuyu alır. Kesilirse kaldığı yerden devam eder.

Kullanım:
    python -X utf8 konu_etiketle.py
    python -X utf8 belge_isle.py        # konuları parçalara işler, ChromaDB'yi yeniden kurar
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import ollama

VERI = Path(__file__).resolve().parent.parent / "veri"
ARTICLES = VERI / "islenmis" / "maddeler.jsonl"
OUT = VERI / "islenmis" / "konu_etiketleri.json"
MODEL = "qwen3:14b"
BATCH = 5  # tek çağrıda etiketlenen madde sayısı
MAX_CHARS = 1200  # madde metninin modele verilen kısmı

KONULAR = {
    "kayit": "kayıt, kayıt yenileme, ders seçimi/kaydı, kayıt silme, ilişik kesme, izinli sayılma (kayıt dondurma)",
    "ders_program": "dersler, AKTS/kredi, ders planı, seçmeli/zorunlu ders, uzaktan eğitim, yaz öğretimi",
    "devam": "derse devam zorunluluğu, devamsızlık",
    "sinav": "sınavlar, ara/final/bütünleme, mazeret sınavı, tek ders sınavı, sınava itiraz",
    "not_basari": "notlar, harf notları, AGNO, başarı değerlendirme, bağıl/mutlak değerlendirme, yüzde on, onur",
    "mezuniyet_diploma": "mezuniyet, diploma, diploma eki, transkript ve diğer belgeler",
    "gecis_program": "yatay/dikey geçiş, çift anadal, yandal, kredi transferi, değişim programları (Erasmus, Farabi)",
    "muafiyet_intibak": "ders muafiyeti, intibak, önceki öğrenmelerin tanınması",
    "hazirlik_yabanci_dil": "yabancı dil hazırlık, yeterlik sınavı, zorunlu yabancı dil dersleri",
    "staj_uygulama": "staj, işletmede mesleki eğitim, uygulamalı eğitim, bitirme çalışması",
    "ucret_katki_payi": "harç, katkı payı, öğrenim ücreti, öğrenciden alınan ücretler",
    "ogrenim_suresi": "öğrenim süresi, azami süre",
    "ogrenci_kabul": "öğrenci kabulü, başvuru, kontenjan, özel öğrenci, uluslararası öğrenci, bilimsel hazırlık",
    "danismanlik": "akademik danışman, tez danışmanı atanması ve görevleri",
    "lisansustu_tez": "tez, tez önerisi, tez izleme, yeterlik sınavı, jüri, tez savunması, proje (tezsiz)",
    "disiplin": "öğrenci veya personel disiplin suçları, cezalar, soruşturma",
    "ders_yuku_ek_ders": "öğretim elemanı ders yükü, ek ders ücreti, ders görevlendirmesi, sınav ücreti",
    "bap_proje": "bilimsel araştırma projeleri (BAP), proje başvurusu, bütçe, rapor, bursiyer",
    "harcirah_gorevlendirme": "harcırah, yolluk, gündelik, yurt içi/dışı görevlendirme",
    "satin_alma_ihale": "ihale, doğrudan temin, satın alma, KDV/tevkifat, fatura",
    "akademik_kadro": "öğretim elemanı atama, doçentlik, profesörlük, kadro, çalışma esasları, personel hakları",
    "teskilat_yonetim": "üniversite/fakülte/enstitü teşkilatı, kurullar, komisyonlar, kalite, yönetim görevleri",
    "genel_hukumler": "amaç, kapsam, dayanak, tanımlar, yürürlük, yürütme, hüküm bulunmayan haller",
    "diger": "yukarıdakilerin hiçbirine uymayan",
}
# Bu başlıklar konu bilgisi taşımaz; modele sorulmadan etiketlenir.
GENERAL_HEADING_RE = re.compile(
    r"^(amaç|kapsam|dayanak|tanımlar|yürürlük|yürütme|amaç ve kapsam|amaç, kapsam|"
    r"yürürlükten kaldırılan|hüküm bulunmayan|yönergede hüküm|yönetmelikte hüküm)", re.IGNORECASE)

SCHEMA = {
    "type": "object",
    "properties": {"etiketler": {"type": "array", "items": {
        "type": "object",
        "properties": {"no": {"type": "integer"}, "konu": {"type": "string", "enum": list(KONULAR)}},
        "required": ["no", "konu"]}}},
    "required": ["etiketler"],
}


def label_batch(batch: list[dict]) -> dict[str, str]:
    topics = "\n".join(f"- {key}: {desc}" for key, desc in KONULAR.items())
    items = "\n\n".join(
        f"#{i}\nBelge: {a['ozellikler']['belge_adi']}\n{a['metin'][:MAX_CHARS]}" for i, a in enumerate(batch, 1))
    prompt = (f"Aşağıdaki mevzuat maddelerinin her birine EN UYGUN tek bir konu seç.\n\nKONULAR:\n{topics}\n\n"
              f"MADDELER:\n{items}\n\nHer madde için no ve konu ver. JSON olarak cevap ver.")
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}], format=SCHEMA,
                           options={"temperature": 0, "num_ctx": 8192, "num_predict": 400}, think=False)
    result = {}
    for item in json.loads(response["message"]["content"]).get("etiketler", []):
        if 1 <= item.get("no", 0) <= len(batch) and item.get("konu") in KONULAR:
            result[batch[item["no"] - 1]["ozellikler"]["madde_id"]] = item["konu"]
    return result


def main() -> int:
    with ARTICLES.open(encoding="utf-8") as input_file:
        articles = [json.loads(line) for line in input_file]
    labels: dict[str, str] = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}

    # Madde başına tek kayıt (uzun maddelerin ilk parçası temsil eder).
    firsts = [a for a in articles if a["ozellikler"]["madde_parca_no"] == 1]
    for article in firsts:
        if GENERAL_HEADING_RE.match(article["ozellikler"]["madde_basligi"] or ""):
            labels.setdefault(article["ozellikler"]["madde_id"], "genel_hukumler")
    todo = [a for a in firsts if a["ozellikler"]["madde_id"] not in labels]
    print(f"Madde: {len(firsts)}, başlıktan etiketlenen/önceden yapılan: {len(firsts) - len(todo)}, "
          f"modele sorulacak: {len(todo)} ({MODEL})")

    start = time.time()
    for index in range(0, len(todo), BATCH):
        batch = todo[index:index + BATCH]
        try:
            labels.update(label_batch(batch))
        except Exception as exc:  # tek bir parti bozulursa diğerleri devam etsin; sonra tekrar denenir
            print(f"  parti {index // BATCH + 1}: HATA {exc}")
        OUT.write_text(json.dumps(labels, ensure_ascii=False, indent=0), encoding="utf-8")
        done = min(index + BATCH, len(todo))
        elapsed = time.time() - start
        print(f"  {done}/{len(todo)}  ({elapsed / 60:.1f} dk, kalan ~{elapsed / done * (len(todo) - done) / 60:.0f} dk)",
              flush=True)

    missing = [a["ozellikler"]["madde_id"] for a in firsts if a["ozellikler"]["madde_id"] not in labels]
    counts: dict[str, int] = {}
    for value in labels.values():
        counts[value] = counts.get(value, 0) + 1
    print("\nDağılım:", dict(sorted(counts.items(), key=lambda kv: -kv[1])))
    print(f"Etiketsiz kalan: {len(missing)} (tekrar çalıştırınca denenir)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
