#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gerçek başarı ölçümü: qwen3:14b hakem, cevabın doğruluğunu ve gösterilen kaynağın cevabı destekleyip
desteklemediğini kontrol eder.

degerlendir_v2.py'nin "doğru kaynak" ölçütü yalnızca beklenen maddeyi arar; aynı kuralı içeren paralel
belgeyi (4734 Kanunu <-> 3(f) Usul ve Esasları, KOÜ <-> YÖK Lisansüstü) gösteren doğru cevaplar yanlış sayılır.
Hakem; soruyu, cevabı, sorunun üretildiği maddeyi ve cevapta gösterilen maddeyi görür.

Kullanım (önce degerlendir_v2.py ile cevaplar_<model>.jsonl üretilmiş olmalı):
    python -X utf8 hakem_degerlendir.py kou-asistan-v2
Çıktı: veri/egitim/hakem_<model>.jsonl (soru soru) + özet ekrana ve hakem_ozet.md'ye
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import ollama

from degerlendir_v2 import ARTICLES, EGITIM, KAYNAK_RE, REFUSAL_RE, answers_path
from kaynak_dogrula import cited_indexes, verify

JUDGE_MODEL = "qwen3:14b"
SCHEMA = {
    "type": "object",
    "properties": {
        "cevap_dogru": {"type": "boolean"},
        "kaynak_destekliyor": {"type": "boolean"},
        "aciklama": {"type": "string"},
    },
    "required": ["cevap_dogru", "kaynak_destekliyor", "aciklama"],
}


def judge(question: str, answer: str, reference: dict[str, Any], cited: list[dict[str, Any]],
          expected: str | None = None) -> dict[str, Any]:
    """expected: test setindeki beklenen kısa cevap. Verilmezse hakem yalnız maddeyi görür; o zaman doğru cevabı
    "maddedeki her koşulu saymadı" diye yanlış sayıyordu (23.10: web_009 cevabı beklenenle aynıyken yanlış)."""
    cited_text = "\n\n".join(f"[{c['ozellikler']['belge_adi']}]\n{c['metin']}" for c in cited) or "(kaynak gösterilmemiş)"
    prompt = (
        "Bir üniversite mevzuat asistanının cevabını değerlendiriyorsun.\n\n"
        f"SORU: {question}\n\n"
        f"ASİSTANIN CEVABI:\n{answer}\n\n"
        f"REFERANS MADDE (soru bu maddeden üretildi):\n[{reference['ozellikler']['belge_adi']}]\n{reference['metin']}\n\n"
        f"CEVAPTA KAYNAK GÖSTERİLEN MADDE:\n{cited_text}\n\n"
        + (f"BEKLENEN KISA CEVAP: {expected}\n\n" if expected else "")
        + "Değerlendir:\n"
        "- cevap_dogru: Cevap soruyu mevzuata göre doğru yanıtlıyor mu? Referans maddeyle ya da kaynak "
        "gösterilen maddeyle çelişen, uydurma veya soruyu cevaplamayan bir cevapsa false. Aynı kuralı başka "
        "bir belgeden (ör. YÖK yönetmeliği) doğru aktarıyorsa true."
        + (" Beklenen kısa cevaptaki bilgiyi aynı anlamda veriyorsa true; maddede geçen ama sorunun sormadığı "
           "ek koşul ve ayrıntıların cevapta olmaması false sebebi değildir. Beklenen cevaptan FAZLA bilgi vermek "
           "de false sebebi değildir (ek bilgi kaynakla çelişmedikçe). Yanlış sayı, tarih, süre ya da "
           "kurum veriyorsa veya başka bir duruma ait kuralı aktarıyorsa false." if expected else "")
        + "\n"
        "- kaynak_destekliyor: Cevaptaki asıl bilgi, kaynak gösterilen maddede gerçekten yazıyor mu?\n"
        "- aciklama: tek cümle.\nJSON ver."
    )
    response = ollama.chat(model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}], format=SCHEMA,
                           think=False, options={"temperature": 0, "num_ctx": 8192})
    try:
        return json.loads(response["message"]["content"])
    except json.JSONDecodeError:
        return {"cevap_dogru": False, "kaynak_destekliyor": False, "aciklama": "hakem çıktısı okunamadı"}


def pct(a: int, b: int) -> str:
    return f"{a}/{b} (%{100 * a / b:.0f})" if b else "-"


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "kou-asistan-v2"
    units = {row["id"]: row for row in map(json.loads, ARTICLES.open(encoding="utf-8"))}
    rows = [json.loads(line) for line in answers_path(model).open(encoding="utf-8")]
    out_path = EGITIM / f"hakem_{model.replace(':', '_')}.jsonl"
    stats = {"toplam": 0, "tam_dogru": 0, "cevap_dogru": 0, "kaynak_destekliyor": 0, "red": 0,
             "yanlis": 0, "kapsam_disi": 0, "dogru_red": 0}
    with out_path.open("w", encoding="utf-8") as output:
        for index, row in enumerate(rows, start=1):
            context = [units[i] for i in row["baglam"]]
            answer, _ = verify(row["cevap"], context)
            kaynak = KAYNAK_RE.search(answer)
            refused = bool(REFUSAL_RE.search(answer)) and not kaynak
            if row["tur"] == "out_of_scope":
                stats["kapsam_disi"] += 1
                stats["dogru_red"] += refused
                continue
            stats["toplam"] += 1
            if refused:
                stats["red"] += 1
                result = {"cevap_dogru": False, "kaynak_destekliyor": False, "aciklama": "bilgi yok dedi"}
            else:
                cited = [context[i] for i in cited_indexes(kaynak.group(0), context)] if kaynak else []
                result = judge(row["soru"], answer, units[row["unit_id"]], cited)
                stats["cevap_dogru"] += bool(result["cevap_dogru"])
                stats["kaynak_destekliyor"] += bool(result["kaynak_destekliyor"])
                stats["tam_dogru"] += bool(result["cevap_dogru"] and result["kaynak_destekliyor"])
                stats["yanlis"] += not result["cevap_dogru"]
            output.write(json.dumps({"soru": row["soru"], "cevap": answer, **result}, ensure_ascii=False) + "\n")
            if index % 25 == 0:
                print(f"  hakem: {index}/{len(rows)}", flush=True)

    n = stats["toplam"]
    summary = "\n".join([
        f"## {model} — hakem ({JUDGE_MODEL}) değerlendirmesi, {n} cevaplanabilir soru",
        "| Ölçüm | Sonuç |", "|---|---|",
        f"| Cevap doğru + kaynak doğru | {pct(stats['tam_dogru'], n)} |",
        f"| Cevap doğru (kaynaktan bağımsız) | {pct(stats['cevap_dogru'], n)} |",
        f"| Gösterilen kaynak cevabı destekliyor | {pct(stats['kaynak_destekliyor'], n)} |",
        f"| Yanlış / eksik cevap | {pct(stats['yanlis'], n)} |",
        f"| \"Bilgi yok\" dedi | {pct(stats['red'], n)} |",
        f"| Kapsam dışı soruyu reddetti | {pct(stats['dogru_red'], stats['kapsam_disi'])} |",
    ])
    (EGITIM / "hakem_ozet.md").write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
