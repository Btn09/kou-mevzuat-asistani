#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model cevabının "Kaynak:" satırını bağlamdaki maddelerle doğrular (eğitim gerektirmez).

1) Kaynak düzeltme: cevap metni hangi bağlam maddesinden alındıysa (kelime ikilisi örtüşmesi) kaynak o
   olmalı. Model bazen cevabı doğru maddeden yazıp komşu maddeyi gösteriyor ("Madde 16" yerine "Madde 26").
   Gösterilen madde cevabı açıkça daha az destekliyorsa kaynak satırı en iyi destekleyen maddeyle değişir.
   Emin olunmayan durumda (düşük örtüşme, çoklu kaynak) cevaba dokunulmaz.
2) Yürürlük uyarısı: cevap yürürlükten kalkmış bir belgeye dayanıyorsa ve model bunu söylemediyse eklenir
   (tarih çalışma anında bugünle karşılaştırılır; ileri tarihli bitişler uyarı üretmez).
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

from raft_format import FIKRA_LINE_RE, format_citation

KAYNAK_LINE_RE = re.compile(r"(?im)^[ \t]*Kaynak\s*:.*$")
MIN_SUPPORT = 0.3  # en iyi madde cevabın en az bu kadarını desteklemiyorsa düzeltme yapılmaz
KEEP_RATIO = 0.5  # gösterilen madde en iyinin bu oranına ulaşıyorsa model kararına dokunulmaz
MODEL_NOTE_RE = re.compile(r"[ \t]*Not: Bu bilgi [^\n]*?yürürlükten kalkan [^\n]*?(?:bakınız|danışınız)\.")
DEPRECATED_NOTE = ("Not: Bu bilgi {tarih} tarihinde yürürlükten kalkan {belge} hükümlerine dayanmaktadır; "
                   "güncel düzenleme için ilgili birime danışınız.")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("İ", "i").replace("I", "ı").lower()).strip()


def bigrams(text: str) -> set[tuple[str, str]]:
    words = [w for w in re.findall(r"\w+", normalize(text)) if len(w) > 1]
    return set(zip(words, words[1:]))


def support(answer: str, text: str) -> float:
    """Cevaptaki kelime ikililerinin ne kadarı metinde geçiyor (0-1)."""
    answer_pairs = bigrams(answer)
    return len(answer_pairs & bigrams(text)) / len(answer_pairs) if answer_pairs else 0.0


def citation_prefix(article: dict[str, Any]) -> str:
    fields = article["ozellikler"]
    return format_citation(fields["madde_etiketi"], [], fields["belge_adi"], fields["madde_basligi"])[len("Kaynak: "):]


def cited_indexes(kaynak_line: str, context: list[dict[str, Any]]) -> list[int]:
    """Kaynak satırında adı geçen bağlam maddeleri ("Madde 1" "Madde 13"ün içinde sayılmaz)."""
    line = normalize(kaynak_line)
    return [index for index, article in enumerate(context)
            if re.search(re.escape(normalize(citation_prefix(article))) + r"(?!\d)", line)]


def best_fikralar(answer: str, article: dict[str, Any]) -> list[int]:
    """Cevabı en çok destekleyen fıkra (kılavuz bölümlerinde numaralı madde yok: boş liste)."""
    text = article["metin"]
    starts = [(m.start(), int(m.group(1))) for m in FIKRA_LINE_RE.finditer(text)]
    if not starts or article["ozellikler"]["madde_etiketi"].startswith("KISIM"):
        return []
    pieces = [(number, text[start:(starts[i + 1][0] if i + 1 < len(starts) else len(text))])
              for i, (start, number) in enumerate(starts)]
    return [max(pieces, key=lambda piece: support(answer, piece[1]))[0]]


def deprecated(article: dict[str, Any], today: dt.date | None = None) -> bool:
    end = article["ozellikler"].get("yururluk_bitis")
    return bool(end) and dt.date.fromisoformat(end) <= (today or dt.date.today())


def verify(answer: str, context: list[dict[str, Any]]) -> tuple[str, str]:
    """(düzeltilmiş cevap, ne yapıldı: "" | "kaynak_duzeltildi" | "yururluk_eklendi" | ikisi)."""
    match = KAYNAK_LINE_RE.search(answer)
    if not match or not context:
        return answer, ""
    body = answer[:match.start()].strip()
    actions = []
    scores = [support(body, article["metin"]) for article in context]
    best = max(range(len(context)), key=lambda i: scores[i])
    cited = cited_indexes(match.group(0), context)
    used = cited
    if not cited:
        # Kaynak satırı hiçbir bağlam maddesine uymuyor (ders sorularında model ders kodunu yazmıyordu: "… – Tip
        # (Lisans) – Tip (Lisans)", 23.10): adı satırda geçen belgelerin maddelerinden cevabı en çok destekleyen
        # seçilir; destek eşiği aranmaz, boşa çıkan kaynak satırından iyidir.
        line = normalize(match.group(0))
        named = [i for i, a in enumerate(context) if normalize(a["ozellikler"]["belge_adi"]) in line] \
            or list(range(len(context)))
        best = max(named, key=lambda i: scores[i])
        fields = context[best]["ozellikler"]
        answer = answer[:match.start()] + format_citation(fields["madde_etiketi"], best_fikralar(body, context[best]),
                                                          fields["belge_adi"], fields["madde_basligi"]) \
            + answer[match.end():]
        used = [best]
        actions.append("kaynak_duzeltildi")
    elif len(cited) <= 1 and scores[best] >= MIN_SUPPORT and best not in cited:
        if not cited or scores[cited[0]] < KEEP_RATIO * scores[best]:
            fields = context[best]["ozellikler"]
            line = format_citation(fields["madde_etiketi"], best_fikralar(body, context[best]),
                                   fields["belge_adi"], fields["madde_basligi"])
            answer = answer[:match.start()] + line + answer[match.end():]
            used = [best]
            actions.append("kaynak_duzeltildi")
    old = [context[i] for i in used if deprecated(context[i])]
    if not old and MODEL_NOTE_RE.search(answer):
        # Eğitim verisinde ileri tarihli bitişi olan Yatay Geçiş Kılavuzu'na da yanlışlıkla KDV notu eklenmişti.
        answer = MODEL_NOTE_RE.sub("", answer)
        actions.append("yanlis_not_silindi")
    if old and "yürürlükten kalk" not in normalize(answer):
        fields = old[0]["ozellikler"]
        date = dt.date.fromisoformat(fields["yururluk_bitis"]).strftime("%d/%m/%Y")
        note = DEPRECATED_NOTE.format(tarih=date, belge=fields["belge_adi"])
        match = KAYNAK_LINE_RE.search(answer)
        answer = f"{answer[:match.start()].rstrip()} {note}\n\n{answer[match.start():].strip()}"
        actions.append("yururluk_eklendi")
    return answer, "+".join(actions)
