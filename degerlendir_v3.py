#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 test seti değerlendirmesi: her öğe rol bilgisi olmadan yeni bir sohbette (rag_v2.Sohbet) çalışır.

- Asistan netleştirme sorarsa test, beklenen birimin adını yazarak cevaplar (gerçek kullanıcı gibi);
  sorulan netleştirme sayısı ayrıca raporlanır.
- 1. geçiş cevapları üretir (veri/test_v3/sonuc_<etiket>.jsonl); 2. geçişte qwen3:14b hakem puanlar.
  --hakemsiz: yalnızca otomatik ölçütler. --kayitli: cevapları yeniden üretmeden puanla.

Kullanım:
    python -X utf8 degerlendir_v3.py v2_baslangic
    python -X utf8 degerlendir_v3.py hat_v3 --hat v3                      # v2 model + v3 arama/sohbet
    python -X utf8 degerlendir_v3.py v3 --hat v3 --model kou-asistan-v3
Çıktı: veri/test_v3/rapor_<etiket>.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from degerlendir_v2 import KAYNAK_RE, REFUSAL_RE, expected_citation, normalize
from hakem_degerlendir import judge
from kaynak_dogrula import cited_indexes

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "veri" / "test_v3"
ARTICLES = ROOT / "veri" / "islenmis" / "maddeler.jsonl"
GENERAL_UNITS = {"tum", "enstitu"}
# Beklenen maddenin başka belgedeki kopyası da doğru sayılır (aynı yönergenin web ve arşiv kopyası, aynı dersin
# başka programdaki kaydı): sistem kopyalardan yetkili olanı seçiyor (rag_v3 bağlam ayıklaması, 19.10).
EQUIVALENT = 0.7


def _words(unit: dict[str, Any]) -> frozenset[str]:
    import re
    from pdf_to_chromadb import normalize_for_matching
    return frozenset(re.findall(r"\w{4,}", normalize_for_matching(unit["metin"])))


def equivalent(expected: dict[str, Any], other: dict[str, Any] | None) -> bool:
    if other is None:
        return False
    if expected["id"] == other["id"]:
        return True
    a, b = _words(expected), _words(other)
    return len(a & b) >= EQUIVALENT * max(1, min(len(a), len(b)))


def birim_reply(birim: str) -> str:
    from rag_v2 import BIRIMLER
    return "fark etmez" if birim in GENERAL_UNITS or not birim else BIRIMLER.get(birim, birim)


def run(model: str, items: list[dict[str, Any]], hat: str = "v2") -> list[dict[str, Any]]:
    if hat == "v3":
        from rag_v3 import Asistan3 as Assistant, Sohbet3 as Sohbet
    else:
        from rag_v2 import Assistant, Sohbet

    assistant = Assistant(model)
    results = []
    for index, item in enumerate(items, start=1):
        sohbet = Sohbet(assistant)  # rol ve birim verilmez
        previous = ""
        turns = item["turlar"]
        position = 0
        while position < len(turns):
            current = turns[position]
            expected = current["beklenen"]
            reply = sohbet.sor(current["mesaj"])
            asked = 0
            skip_next = False
            if expected["tip"] == "netlestirme_veya_cevap":
                if reply["tur"] == "netlestirme":
                    asked = 1
                    position += 1  # sonraki tur zaten birim adını yazıyor
                    reply = sohbet.sor(turns[position]["mesaj"])
                # Doğrudan cevap verdiyse birim adını yazan tur atlanır.
                else:
                    skip_next = True
            while reply["tur"] == "netlestirme" and asked < 2:  # beklenmeyen netleştirme: birimini söyle
                asked += 1
                reply = sohbet.sor(birim_reply(expected["birim"]))
            results.append({
                "id": item["id"], "tur": item["tur"], "senaryo": item.get("senaryo", ""), "kitle": item["kitle"],
                "birim_soruda": item.get("birim_soruda"), "tur_no": position + 1,
                "soru": current["mesaj"], "onceki_soru": previous, "beklenen": expected,
                "netlestirme_sayisi": asked, "cevap_turu": reply["tur"], "cevap": reply["metin"],
                "baglam": [a["id"] for a in reply["baglam"]],
            })
            previous = current["mesaj"]
            position += 2 if skip_next else 1
        if index % 10 == 0:
            print(f"  {index}/{len(items)}", flush=True)
    return results


def score(results: list[dict[str, Any]], units: dict[str, dict[str, Any]], use_judge: bool) -> list[dict[str, Any]]:
    for index, row in enumerate(results, start=1):
        answer = row["cevap"]
        kaynak = KAYNAK_RE.search(answer)
        refused = bool(REFUSAL_RE.search(answer)) and not kaynak
        row["red"] = refused
        unit_id = row["beklenen"]["unit_id"]
        if not unit_id:  # kapsam dışı
            row["dogru"] = refused
            continue
        unit = units.get(unit_id)
        if unit is None:  # beklenen madde veriden silinmiş (ör. KDV Tebliği 117, 22.10): öğe puanlanmaz
            row["dogru"] = row["arama"] = row["kaynak_tam"] = False
            row["hakem_not"] = "beklenen madde veride yok"
            continue
        row["arama"] = any(equivalent(unit, units.get(i)) for i in row["baglam"])
        same = [unit] + [units[i] for i in row["baglam"] if i in units and i != unit_id and equivalent(unit, units[i])]
        row["kaynak_tam"] = bool(kaynak) and any(normalize(expected_citation(u)) in normalize(kaynak.group(1))
                                                 for u in same)
        if use_judge and not refused:
            context = [units[i] for i in row["baglam"] if i in units]  # kayıtlı sonuçta sonradan silinen madde olabilir
            cited = [context[i] for i in cited_indexes(kaynak.group(0), context)] if kaynak else []
            question = (f"(Sohbetteki önceki soru: {row['onceki_soru']}) " if row["onceki_soru"] else "") + row["soru"]
            verdict = judge(question, answer, unit, cited, row["beklenen"].get("cevap_ozeti"))
            row["hakem_cevap"] = bool(verdict["cevap_dogru"])
            row["hakem_kaynak"] = bool(verdict["kaynak_destekliyor"])
            row["hakem_not"] = verdict.get("aciklama", "")
        elif use_judge:
            row["hakem_cevap"] = row["hakem_kaynak"] = False
            row["hakem_not"] = "bilgi yok dedi"
        row["dogru"] = row.get("hakem_cevap", row["kaynak_tam"]) and row.get("hakem_kaynak", True)
        if index % 25 == 0:
            print(f"  hakem: {index}/{len(results)}", flush=True)
    return results


def pct(a: int, b: int) -> str:
    return f"{a}/{b} (%{100 * a / b:.0f})" if b else "-"


def report(label: str, results: list[dict[str, Any]], use_judge: bool) -> str:
    def block(name: str, rows: list[dict[str, Any]]) -> str:
        answerable = [r for r in rows if r["beklenen"]["unit_id"]]
        n = len(answerable)
        line = (f"| {name} | {n} | {pct(sum(r['arama'] for r in answerable), n)} | "
                f"{pct(sum(r['kaynak_tam'] for r in answerable), n)} | ")
        line += (f"{pct(sum(r.get('hakem_cevap', False) for r in answerable), n)} | "
                 f"{pct(sum(r['dogru'] for r in answerable), n)} | " if use_judge else "- | - | ")
        line += f"{sum(r['red'] for r in answerable)} |"
        return line

    single = [r for r in results if r["tur"] == "tek"]
    multi = [r for r in results if r["tur"] == "cok_turlu"]
    oos = [r for r in results if r["tur"] == "kapsam_disi"]
    lines = [f"## Test sonucu: {label}", "",
             "| Grup | Soru | Arama (madde geldi) | Doğru kaynak (birebir) | Hakem: cevap doğru | "
             "Cevap + kaynak doğru | \"Bilgi yok\" |", "|---|---|---|---|---|---|---|",
             block("Tümü (cevaplanabilir)", single + multi),
             block("Tek soru — hoca/personel", [r for r in single if r["kitle"] == "personel"]),
             block("Tek soru — öğrenci", [r for r in single if r["kitle"] == "ogrenci"]),
             block("Çok turlu — devam sorusu", [r for r in multi if r["senaryo"] == "devam"]),
             block("Çok turlu — netleştirme", [r for r in multi if r["senaryo"] == "netlestirme"]),
             block("Çok turlu — konu değişimi", [r for r in multi if r["senaryo"] == "konu_degisimi"]), ""]
    named = [r for r in single if r["birim_soruda"] is True]
    unnamed = [r for r in single if r["birim_soruda"] is False]
    general = [r for r in single if r["birim_soruda"] is None]
    lines += ["| Netleştirme | Oran |", "|---|---|",
              f"| Birimi belli olmayan fakülte sorusunda sordu | {pct(sum(r['netlestirme_sayisi'] > 0 for r in unnamed), len(unnamed))} |",
              f"| Birim soruda yazılıyken yine sordu (gereksiz) | {pct(sum(r['netlestirme_sayisi'] > 0 for r in named), len(named))} |",
              f"| Genel belge sorusunda sordu | {pct(sum(r['netlestirme_sayisi'] > 0 for r in general), len(general))} |",
              f"| Kapsam dışı soruyu reddetti | {pct(sum(r['dogru'] for r in oos), len(oos))} |", ""]
    return "\n".join(lines)


def main() -> int:
    args = sys.argv[1:]
    model = "kou-asistan-v2"
    if "--model" in args:
        position = args.index("--model")
        model = args[position + 1]
        del args[position:position + 2]
    hat = "v2"
    if "--hat" in args:
        position = args.index("--hat")
        hat = args[position + 1]
        del args[position:position + 2]
    test_file = "test_seti.jsonl"
    if "--set" in args:  # ör. --set web -> test_seti_web.jsonl
        position = args.index("--set")
        test_file = f"test_seti_{args[position + 1]}.jsonl"
        del args[position:position + 2]
    if hat == "v3" and "--model" not in sys.argv:
        # v3 hattında varsayılan, sunucunun kullandığı model (23.10: b4 ölçümleri yanlışlıkla v2 ile yapılmıştı)
        from rag_v3 import CHAT_MODEL
        model = CHAT_MODEL
    use_judge = "--hakemsiz" not in args
    saved = "--kayitli" in args
    args = [a for a in args if not a.startswith("--")]
    label = args[0] if args else model
    units = {row["id"]: row for row in map(json.loads, ARTICLES.open(encoding="utf-8"))}
    courses = ARTICLES.with_name("dersler.jsonl")  # ders soruları (yeni test seti) ders birimlerini bekler
    if courses.exists():
        units.update({row["id"]: row for row in map(json.loads, courses.open(encoding="utf-8"))})
    items = [json.loads(line) for line in (TEST_DIR / test_file).open(encoding="utf-8")]
    out = TEST_DIR / f"sonuc_{label}.jsonl"
    if saved:
        results = [json.loads(line) for line in out.open(encoding="utf-8")]
    else:
        print(f"{len(items)} test öğesi, model {model}, arama hattı {hat}", flush=True)
        results = run(model, items, hat)
    results = score(results, units, use_judge)
    with out.open("w", encoding="utf-8") as output:
        output.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
    text = report(label, results, use_judge)
    (TEST_DIR / f"rapor_{label}.md").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
