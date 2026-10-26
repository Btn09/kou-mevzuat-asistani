#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KOÜ mevzuat asistanı web arayüzü (FastAPI) — v3: hoca ve öğrenci için tek, hafızalı sohbet.

Kullanım:
    python -X utf8 web_app.py
    -> tarayıcıda http://127.0.0.1:8000

- Rol sorulmaz. Her sekme kendi sohbetini açar (/api/sohbet/yeni); hafıza (önceki sorular, öğrenilen birim)
  yalnızca sunucu belleğinde ve o sohbet açıkken tutulur. Sekme kapanınca (/api/sohbet/kapat) ya da
  OTURUM_ZAMAN_ASIMI boyunca kullanılmazsa silinir; sohbet geçmişi diske yazılmaz.
- Asistan belirsiz soruda (hangi fakülte? proje alımı mı?) seçenek sunar; arayüz bunları buton gösterir.
- 👍/👎: yalnızca kullanıcı tıklarsa o tek soru-cevap geri_bildirim.jsonl'ye yazılır (iyileştirme için).
Ollama açık olmalı (CHAT_MODEL ve bge-m3).
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import ollama
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import belge_bilgisi
from belge_isle import EMBEDDING_MODEL
from kaynak_dogrula import KAYNAK_LINE_RE, cited_indexes
from rag_v3 import CHAT_MODEL, Asistan3, Sohbet3

# ------------------------- AYARLAR -------------------------
HOST = "127.0.0.1"  # Ağdaki diğer cihazlardan erişim için "0.0.0.0"
PORT = 8000
PROJECT_DIR = Path(__file__).resolve().parent
FEEDBACK_LOG = PROJECT_DIR / "geri_bildirim.jsonl"
# "Bilgi yok" denen sorular (eksik konuları bulmak için; kullanıcı istedi 19.10). Yalnız soru metni ve gün tutulur:
# oturum, sohbet geçmişi ya da kişi bilgisi YOK. Rapor: python -X utf8 eksik_konular.py
UNANSWERED_LOG = PROJECT_DIR / "bilgi_yok.jsonl"
# Cevabın altında kaynağın tam metni: en az bu kadar numaralı adım, kaynak en fazla bu uzunlukta ve cevap
# kaynağın bu oranından kısaysa (uzun kanun maddelerinin tamamı basılmasın).
DETAIL_MIN_STEPS = 3
DETAIL_MAX_CHARS = 2500
DETAIL_RATIO = 0.6
MAX_MESSAGE_CHARS = 500
OTURUM_ZAMAN_ASIMI = 2 * 3600  # kullanılmayan sohbet bu kadar saniye sonra bellekten silinir
MAX_OTURUM = 200

app = FastAPI(title="KOÜ Mevzuat Asistanı")
asistan = Asistan3(CHAT_MODEL)
sessions: dict[str, dict[str, Any]] = {}  # oturum -> {"sohbet": Sohbet3, "son": zaman}
sessions_lock = threading.Lock()
model_lock = threading.Lock()  # tek GPU: istekler sırayla işlenir


class Message(BaseModel):
    oturum: str = Field(min_length=1, max_length=64)
    mesaj: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class Feedback(BaseModel):
    soru: str = Field(max_length=MAX_MESSAGE_CHARS)
    cevap: str = Field(max_length=4000)
    begeni: bool
    not_: str = Field(default="", max_length=500, alias="not")


def cleanup() -> None:
    now = time.time()
    with sessions_lock:
        for key in [k for k, v in sessions.items() if now - v["son"] > OTURUM_ZAMAN_ASIMI]:
            del sessions[key]


def article_view(article: dict[str, Any]) -> dict[str, Any]:
    fields = article["ozellikler"]
    return {"id": article["id"], "belge": fields["belge_adi"], "madde": fields["madde_etiketi"],
            "baslik": fields.get("madde_basligi", ""), "bolum": fields.get("bolum_basligi", ""),
            "sayfa": fields.get("sayfa"), "not": fields.get("belge_notu", ""), "metin": article["metin"],
            **belge_bilgisi.of(fields["belge_id"])}  # orijinal belge bağlantısı ve tarih etiketi


@app.get("/")
def index() -> FileResponse:
    # no-cache: arayüz güncellenince tarayıcı eski sayfayı önbellekten göstermesin
    return FileResponse(PROJECT_DIR / "web" / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/saglik")
def health() -> dict[str, Any]:
    """Arayüz açılışta bunu çağırır; Ollama kapalıysa kullanıcıya söyler."""
    try:
        installed = {model.model.split(":")[0] for model in ollama.list().models}
    except Exception:
        return {"hazir": False, "mesaj": "Ollama'ya bağlanılamadı. Ollama uygulamasını açın."}
    missing = [name for name in (CHAT_MODEL, EMBEDDING_MODEL) if name.split(":")[0] not in installed]
    if missing:
        return {"hazir": False, "mesaj": f"Ollama'da eksik model: {', '.join(missing)}"}
    return {"hazir": True, "mesaj": "Hazır"}


@app.post("/api/sohbet/yeni")
def new_session() -> dict[str, str]:
    cleanup()
    with sessions_lock:
        if len(sessions) >= MAX_OTURUM:  # en eski sohbet düşer
            del sessions[min(sessions, key=lambda k: sessions[k]["son"])]
        key = uuid.uuid4().hex
        sessions[key] = {"sohbet": Sohbet3(asistan), "son": time.time()}
    return {"oturum": key}


@app.post("/api/sohbet/kapat")
async def close_session(request: Request) -> dict[str, bool]:
    """Sekme kapanırken (sendBeacon) çağrılır: sohbet hafızası silinir."""
    try:
        key = json.loads(await request.body()).get("oturum", "")
    except Exception:
        key = ""
    with sessions_lock:
        removed = sessions.pop(key, None) is not None
    return {"silindi": removed}


@app.post("/api/sohbet")
def chat(message: Message) -> dict[str, Any]:
    text = message.mesaj.strip()
    with sessions_lock:
        session = sessions.get(message.oturum)
        if session is None:
            raise HTTPException(status_code=410, detail="Sohbet süresi doldu; yeni sohbet başlatın.")
        session["son"] = time.time()
    try:
        with model_lock:
            reply = session["sohbet"].sor(text)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model şu an cevap veremiyor: {exc}") from exc

    if reply["tur"] == "netlestirme":
        lines = reply["metin"].split("\n")
        return {"tur": "netlestirme", "cevap": lines[0], "kaynak": "", "kaynak_maddeler": [],
                "secenekler": [{"anahtar": k, "ad": ad} for k, ad in reply["secenekler"]], "reddedildi": False}
    answer, context = reply["metin"], reply["baglam"]
    match = KAYNAK_LINE_RE.search(answer)
    body = answer[:match.start()].strip() if match else answer.strip()
    kaynak = match.group(0).split(":", 1)[1].strip() if match else ""
    cited = [context[i] for i in cited_indexes(match.group(0), context)] if match else []
    views = [article_view(a) for a in cited]
    if not views and "tablosu" in kaynak.lower() and "ücret" in kaynak.lower():  # harç karar ağacı: tutar doğrudan tablodan geldi
        fee = next((v for k, v in belge_bilgisi.table().items() if "harc_ucretleri" in k), None)
        if fee:
            note = body.split("📌", 1)[-1].strip() if "📌" in body else body
            views = [{"id": "", "belge": kaynak, "madde": "", "baslik": "", "bolum": "", "sayfa": None, "not": "",
                      "metin": note, **fee}]
    if not match:  # cevap kaynaksız: "bilgi yok" (red) cevabı
        with sessions_lock, UNANSWERED_LOG.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps({"gun": datetime.now().date().isoformat(), "soru": text}, ensure_ascii=False) + "\n")
    return {"tur": "cevap", "cevap": body, "kaynak": kaynak, "kaynak_maddeler": views,
            "secenekler": [], "reddedildi": not match, "ayrinti": details_of(body, cited)}


def details_of(body: str, cited: list[dict[str, Any]]) -> str:
    """Model kısa bir özet verdiyse ve kaynak kısa bir prosedür/listeyse kaynağın tam metni cevabın altında
    gösterilir (kulüp kurma sorusunda model 6 adımın 2'sini yazdı; adımların tamamı kaynakta vardı).
    Metin kaynaktan aynen gelir; model bir şey uydurmaz."""
    if not cited:
        return ""
    text = re.sub(r"^KISIM \d+ –[^\n]*\n", "", cited[0]["metin"].strip())  # iç bölüm başlığı gösterilmez
    steps = len(re.findall(r"(?:^|\s)(?:\d{1,2}[.)]|\([a-zçğıöşü1-9]\)|[a-zçğıöşü]\))\s", text))
    if DETAIL_MIN_STEPS <= steps and len(text) <= DETAIL_MAX_CHARS and len(body) < DETAIL_RATIO * len(text):
        # Numaralı adımları ayrı satırlara böl ("... gerekir: 3. Bir kulübün ...")
        return re.sub(r"\s+(?=(?:\d{1,2}[.)]|\([a-zçğıöşü1-9]\))\s)", "\n", text)
    return ""


@app.post("/api/geri-bildirim")
def feedback(item: Feedback) -> dict[str, bool]:
    record = {"zaman": datetime.now().isoformat(timespec="seconds"), "soru": item.soru, "cevap": item.cevap,
              "begeni": item.begeni, "not": item.not_}
    with sessions_lock, FEEDBACK_LOG.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"kaydedildi": True}


if __name__ == "__main__":
    print(f"KOÜ Mevzuat Asistanı ({CHAT_MODEL}): http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
