#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AKTS ders dosyalarını (akts_tara.py) ayrı bir koleksiyona işler: her ders bir bağlam birimi.

Ders soruları ("Yazılım Mühendisliği 3. sınıf güz dersleri", "BLM301 kaç AKTS", "Veri Yapıları dersinin ön koşulu")
ana mevzuat aramasını boğmasın diye dersler ayrı koleksiyonda (kou_dersler); rag_v3 ders sorusu sezdiğinde burada da arar.
Çıktı: veri/islenmis/dersler.jsonl (birimler), veri/islenmis/ders_parcalari.jsonl, ChromaDB koleksiyonu kou_dersler.

Kullanım:  python -X utf8 dersler_isle.py
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

from belge_isle import CHROMA_PATH, EMBEDDING_MODEL, embed_with_retry

ROOT = Path(__file__).resolve().parent.parent
AKTS = ROOT / "veri" / "web" / "akts"
OUT = ROOT / "veri" / "islenmis"
COLLECTION = "kou_dersler"
MAX_CHARS = 3000
BATCH = 32
HEADING_RE = re.compile(r"^([A-ZÇĞİÖŞÜ0-9]{2,6}\s?[0-9]{2,4}[A-Z]?) (.+)$")


def courses_in(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    head, _, body = raw.partition("\n\n")
    # akts_tara.py başlıkları str.title() ile yazdı: "Edebi̇yat" (i + birleşik nokta) düzeltilir
    title = re.search(r"BAŞLIK: (.*)", head).group(1).replace("Ders Bilgileri – ", "").replace("i̇", "i")
    url = re.search(r"KAYNAK: (.*)", head).group(1)
    units, current = [], None
    for line in body.split("\n"):
        match = HEADING_RE.match(line.strip())
        if match and line.strip().isupper():
            current = {"kod": match.group(1).replace(" ", ""), "ad": match.group(2).strip(), "satirlar": []}
            units.append(current)
        elif current is not None and line.strip():
            current["satirlar"].append(line.strip())
    out = []
    for unit in units:
        fields = dict(l.split(": ", 1) for l in unit["satirlar"] if ": " in l and len(l) < 120)
        text = "\n".join(unit["satirlar"])
        header = f"{unit['kod']} {unit['ad']} ({title})"
        parts = [text[i:i + MAX_CHARS] for i in range(0, len(text), MAX_CHARS)] or [""]
        for number, part in enumerate(parts, start=1):
            unit_id = f"{path.stem}__{unit['kod']}#{number}"
            out.append({"id": unit_id, "metin": f"{header}\n{part}", "ozellikler": {
                "belge_id": f"aktsders_{path.stem}", "belge_adi": f"AKTS Ders Bilgileri – {title}", "kurum": "KOÜ",
                "hiyerarsi": 6, "hedef_kitle": "herkes", "birim": "tum", "konu": "ders_program",
                "madde_id": f"{path.stem}__{unit['kod']}", "madde_no": unit["kod"], "madde_parca_no": number,
                "atiflar": [], "yururluk_tarihi": "", "yururluk_bitis": "", "madde_etiketi": unit["kod"],
                "madde_basligi": unit["ad"].title() if unit["ad"].isupper() else unit["ad"], "bolum_basligi": title,
                "sayfa": 0, "son_degisiklik": "", "esanlamlilar": "", "belge_notu": "",
                "ders_kodu": unit["kod"], "akts": fields.get("AKTS Kredisi", ""), "donem": fields.get("Dersin Verildiği Dönem", ""),
                "yil": fields.get("Dersin Yılı", ""), "kaynak_url": url}})
    return out


def semester_summaries(units: list[dict]) -> list[dict]:
    """Program + sınıf + dönem başına ders listesi: "3. sınıf güz dersleri neler?" sorusunda tek tek ders
    birimleri yarışıyor, model listeden yalnız bir dersi söylüyordu (18.10)."""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for unit in units:
        fields = unit["ozellikler"]
        if fields["madde_parca_no"] != 1 or not fields["yil"].isdigit() or fields["donem"] not in ("Güz", "Bahar"):
            continue
        groups.setdefault((fields["belge_id"], fields["yil"], fields["donem"]), []).append(unit)
    out = []
    for (belge, year, term), members in sorted(groups.items()):
        first = members[0]["ozellikler"]
        program = first["bolum_basligi"]
        lines = {"Zorunlu": [], "Seçmeli": []}
        for unit in sorted(members, key=lambda u: u["ozellikler"]["ders_kodu"]):
            kind = "Seçmeli" if re.search(r"Dersin Türü: Seçmeli", unit["metin"]) else "Zorunlu"
            fields = unit["ozellikler"]
            lines[kind].append(f"{fields['ders_kodu']} {fields['madde_basligi']} ({fields['akts'] or '?'} AKTS)")
        body = "\n".join(f"{kind} dersler: " + "; ".join(items) for kind, items in lines.items() if items)
        title = f"{program} – {year}. sınıf {term} dönemi dersleri"
        unit_id = f"{belge.removeprefix('aktsders_')}__OZET_{year}_{term}"
        out.append({"id": f"{unit_id}#1", "metin": f"{title} (AKTS bilgi paketi)\n{body}", "ozellikler": {
            **{k: first[k] for k in first if k not in ("ders_kodu", "akts", "donem", "yil")},
            "madde_id": unit_id, "madde_no": f"{year}. sınıf {term}", "madde_parca_no": 1,
            "madde_etiketi": f"{year}. sınıf {term} dönemi", "madde_basligi": f"{year}. sınıf {term} dönemi dersleri",
            "ders_kodu": "", "akts": "", "donem": term, "yil": year}})
    # Program başına tek özet ("X bölümünün dersleri neler?"): zorunlu dersler dönem dönem, seçmeliler sayıyla
    programs: dict[str, list[tuple[str, str, list[dict]]]] = {}
    for (belge, year, term), members in sorted(groups.items(), key=lambda g: (g[0][0], g[0][1], g[0][2] != "Güz")):
        programs.setdefault(belge, []).append((year, term, members))
    for belge, semesters in programs.items():
        first = semesters[0][2][0]["ozellikler"]
        lines = []
        for year, term, members in semesters:
            required = [f"{u['ozellikler']['ders_kodu']} {u['ozellikler']['madde_basligi']}"
                        for u in sorted(members, key=lambda u: u["ozellikler"]["ders_kodu"])
                        if not re.search(r"Dersin Türü: Seçmeli", u["metin"])]
            electives = len(members) - len(required)
            lines.append(f"{year}. sınıf {term}: " + "; ".join(required)
                         + (f" (+{electives} seçmeli ders)" if electives else ""))
        unit_id = f"{belge.removeprefix('aktsders_')}__OZET_PROGRAM"
        out.append({"id": f"{unit_id}#1", "metin": f"{first['bolum_basligi']} – tüm dersler (sınıf ve döneme göre, "
                                                   f"AKTS bilgi paketi)\n" + "\n".join(lines), "ozellikler": {
            **{k: first[k] for k in first if k not in ("ders_kodu", "akts", "donem", "yil")},
            "madde_id": unit_id, "madde_no": "ders listesi", "madde_parca_no": 1, "madde_etiketi": "Ders listesi",
            "madde_basligi": "Programın dersleri", "ders_kodu": "", "akts": "", "donem": "", "yil": ""}})
    return out


def main() -> int:
    import chromadb

    units = [u for path in sorted(AKTS.glob("dersler__*.txt")) for u in courses_in(path)]
    units += semester_summaries(units)
    seen, unique = set(), []
    for unit in units:
        if unit["id"] not in seen:
            seen.add(unit["id"])
            unique.append(unit)
    with (OUT / "dersler.jsonl").open("w", encoding="utf-8") as output:
        output.writelines(json.dumps(u, ensure_ascii=False) + "\n" for u in unique)
    chunks = [{"id": u["id"].replace("#", "__p"), "metin": u["metin"], "ozellikler": {
        **{k: v for k, v in u["ozellikler"].items() if k in ("belge_id", "belge_adi", "kurum", "hiyerarsi", "hedef_kitle",
                                                            "birim", "konu", "madde_id", "madde_no", "madde_parca_no",
                                                            "yururluk_tarihi", "yururluk_bitis")},
        "atiflar": "", "fikra_no": 0, "parca_turu": "ders"}} for u in unique]
    with (OUT / "ders_parcalari.jsonl").open("w", encoding="utf-8") as output:
        output.writelines(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks)
    print(f"Ders birimi: {len(unique)}, parça: {len(chunks)}", flush=True)

    import ollama
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    if "--artimli" in sys.argv[1:]:  # yalnız koleksiyonda olmayan parçalar eklenir (ör. dönem özetleri)
        collection = client.get_collection(COLLECTION)
        have = set(collection.get(include=[])["ids"])
        todo = [c for c in chunks if c["id"] not in have]
        for start in range(0, len(todo), BATCH):
            batch = todo[start:start + BATCH]
            collection.add(ids=[c["id"] for c in batch], documents=[c["metin"] for c in batch],
                           metadatas=[c["ozellikler"] for c in batch],
                           embeddings=embed_with_retry(ollama, [c["metin"] for c in batch]))
        print(f"Artımlı: {len(todo)} parça eklendi; ChromaDB: {collection.count()} ders parçası")
        return 0
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    for start in range(0, len(chunks), BATCH):
        batch = chunks[start:start + BATCH]
        vectors = embed_with_retry(ollama, [c["metin"] for c in batch])
        collection.add(ids=[c["id"] for c in batch], documents=[c["metin"] for c in batch],
                       metadatas=[c["ozellikler"] for c in batch], embeddings=vectors)
        if start % 3200 == 0:
            print(f"  Embedding: {start + len(batch)}/{len(chunks)}", flush=True)
    print(f"ChromaDB: {collection.count()} ders parçası -> {CHROMA_PATH} ({COLLECTION})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
