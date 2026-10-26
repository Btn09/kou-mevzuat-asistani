#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Çok belgeli sistemi test setinde ölçer: test soruları canlı akıştan geçer (rag_v2 arama + model).

Test maddeleri eğitimde hiç görülmedi. Ölçülenler:
  - arama isabeti   : sorunun ait olduğu madde bağlama geldi mi
  - doğru kaynak    : cevabın "Kaynak:" satırı doğru belge + maddeyi gösteriyor mu
  - gereksiz red    : madde bağlamdayken "bilgi yok" demek
  - kapsam dışı red : mevzuat dışı soruyu reddetmek
  - yürürlük notu   : yürürlükten kalkmış belgeye dayanan cevapta bunu söylemek

Kullanım:
    python -X utf8 degerlendir_v2.py kou-asistan-v2 trendyol-8b-ham qwen3:8b
    python -X utf8 degerlendir_v2.py --limit 50 kou-asistan-v2
    python -X utf8 degerlendir_v2.py --kayitli kou-asistan-v2   # kayıtlı cevapları yeniden puanla (LLM yok)
Çıktı: veri/egitim/degerlendirme_<model>.txt (soru soru) ve özet tablo (ekrana + degerlendirme_ozet.md)
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Any

from kaynak_dogrula import deprecated, verify
from rag_v2 import Assistant
from raft_format import format_citation

EGITIM = Path(__file__).resolve().parent.parent / "veri" / "egitim"
TEST = EGITIM / "raft_test.jsonl"
ARTICLES = Path(__file__).resolve().parent.parent / "veri" / "islenmis" / "maddeler.jsonl"
SEED = 42
DEFAULT_LIMIT = 200  # cevaplanabilir soru sayısı (kapsam dışıların hepsi ayrıca)

KAYNAK_RE = re.compile(r"Kaynak\s*:(.*)", re.IGNORECASE)
REFUSAL_RE = re.compile(
    r"(yer al(m[ıi]yor|mama|maz)|bulunm(a|u)|bilgi (yok|bulun|içermiyor)|belirtilme|içermemekte|danış)", re.IGNORECASE)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("İ", "i").replace("I", "ı").lower()).strip()


def expected_citation(unit: dict[str, Any]) -> str:
    """Fıkra numarası olmadan beklenen kaynak: 'KOÜ ... Yönetmeliği, Madde 22'."""
    fields = unit["ozellikler"]
    return format_citation(fields["madde_etiketi"], [], fields["belge_adi"], fields["madde_basligi"])[len("Kaynak: "):]


def load_cases(limit: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in TEST.open(encoding="utf-8") if line.strip()]
    answerable = [r for r in rows if r["meta"]["type"] in ("oracle", "no_oracle")]
    out_of_scope = [r for r in rows if r["meta"]["type"] == "out_of_scope"]
    rng = random.Random(SEED)
    rng.shuffle(answerable)
    # no_oracle da gerçek aramada cevaplanabilir: o madde eğitimde bağlamdan çıkarılmıştı, burada değil.
    return answerable[:limit] + out_of_scope


def generate(model: str, cases: list[dict[str, Any]], units: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Cevapları üretir (kaynak doğrulaması kapalı) ve cevaplar_<model>.jsonl'ye kaydeder."""
    assistant = Assistant(model)
    rows = []
    for index, case in enumerate(cases, start=1):
        meta = case["meta"]
        unit = units.get(meta["unit_id"])
        # Arayüz senaryosu: personel "personelim" seçer (birim seçmez); öğrenci kendi birimini seçer.
        fields = unit["ozellikler"] if unit else {}
        kitle = {"personel": "personel", "ogrenci": "ogrenci"}.get(fields.get("hedef_kitle"))
        birim = fields.get("birim") if kitle == "ogrenci" and fields.get("birim") != "tum" else None
        answer, context = assistant.answer(meta["question"], kitle=kitle, birim=birim, dogrula=False)
        rows.append({"soru": meta["question"], "tur": meta["type"], "unit_id": meta["unit_id"],
                     "cevap": answer, "baglam": [c["id"] for c in context]})
        if index % 25 == 0:
            print(f"  {model}: {index}/{len(cases)}", flush=True)
    with answers_path(model).open("w", encoding="utf-8") as output:
        output.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    return rows


def answers_path(model: str) -> Path:
    return EGITIM / f"cevaplar_{model.replace(':', '_')}.jsonl"


def score(name: str, rows: list[dict[str, Any]], units: dict[str, dict[str, Any]], dogrula: bool) -> dict[str, Any]:
    stats = {"model": name, "cevaplanabilir": 0, "arama_isabet": 0, "dogru_kaynak": 0, "gereksiz_red": 0,
             "kapsam_disi": 0, "dogru_red": 0, "yururluk_gerekli": 0, "yururluk_notu": 0, "duzeltilen": 0}
    lines = []
    for index, row in enumerate(rows, start=1):
        unit = units.get(row["unit_id"])
        context = [units[i] for i in row["baglam"]]
        answer, action = verify(row["cevap"], context) if dogrula else (row["cevap"], "")
        stats["duzeltilen"] += "kaynak" in action
        kaynak = KAYNAK_RE.search(answer)
        refused = bool(REFUSAL_RE.search(answer)) and not kaynak
        if row["tur"] == "out_of_scope":
            stats["kapsam_disi"] += 1
            stats["dogru_red"] += refused
            verdict = "doğru red" if refused else "UYDURDU"
        else:
            stats["cevaplanabilir"] += 1
            hit = unit["id"] in row["baglam"]
            correct = bool(kaynak) and normalize(expected_citation(unit)) in normalize(kaynak.group(1))
            stats["arama_isabet"] += hit
            stats["dogru_kaynak"] += correct
            stats["gereksiz_red"] += refused and hit
            if any(deprecated(c) for c in context if c["id"] == unit["id"]):
                stats["yururluk_gerekli"] += 1
                stats["yururluk_notu"] += "yürürlükten kalk" in answer.lower()
            verdict = f"beklenen: {expected_citation(unit)} | getirildi: {'evet' if hit else 'HAYIR'} | " \
                      f"{'DOĞRU' if correct else 'yanlış'} kaynak" + (f" | {action}" if action else "")
        brought = " || ".join(f"{c['ozellikler']['belge_adi'][:45]} {c['ozellikler']['madde_etiketi']}" for c in context)
        lines.append(f"[{index}] {row['soru']}\n    -> {answer.replace(chr(10), ' ⏎ ')[:400]}\n    ({verdict})"
                     f"\n    bağlam: {brought}")
    (EGITIM / f"degerlendirme_{name.replace(':', '_')}.txt").write_text("\n".join(lines), encoding="utf-8")
    return stats


def pct(a: int, b: int) -> str:
    return f"{a}/{b} (%{100 * a / b:.0f})" if b else "-"


def main() -> int:
    args = sys.argv[1:]
    limit = DEFAULT_LIMIT
    if "--limit" in args:
        position = args.index("--limit")
        limit = int(args[position + 1])
        del args[position:position + 2]
    saved = "--kayitli" in args  # cevapları yeniden üretme, cevaplar_<model>.jsonl'yi puanla
    if saved:
        args.remove("--kayitli")
    models = args or ["kou-asistan-v2"]
    units = {row["id"]: row for row in map(json.loads, ARTICLES.open(encoding="utf-8"))}
    cases = load_cases(limit)
    print(f"Test: {len(cases)} soru ({limit} cevaplanabilir + kapsam dışı)")
    results = []
    for model in models:
        if saved:
            rows = [json.loads(line) for line in answers_path(model).open(encoding="utf-8")]
        else:
            rows = generate(model, cases, units)
        results.append(score(model, rows, units, dogrula=False))
        results.append(score(f"{model}+dogrulama", rows, units, dogrula=True))

    table = ["| Model | Arama isabeti | Doğru kaynak | Gereksiz red | Kapsam dışı red | Yürürlük notu | Düzeltilen kaynak |",
             "|---|---|---|---|---|---|---|"]
    for s in results:
        table.append(f"| {s['model']} | {pct(s['arama_isabet'], s['cevaplanabilir'])} | "
                     f"{pct(s['dogru_kaynak'], s['cevaplanabilir'])} | {s['gereksiz_red']} | "
                     f"{pct(s['dogru_red'], s['kapsam_disi'])} | {pct(s['yururluk_notu'], s['yururluk_gerekli'])} | "
                     f"{s['duzeltilen']} |")
    summary = "\n".join(table)
    (EGITIM / "degerlendirme_ozet.md").write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
