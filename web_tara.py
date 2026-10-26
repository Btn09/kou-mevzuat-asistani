#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KOÜ web sitesindeki (PDF olmayan) mevzuat/işlem sayfalarını toplar.

KOÜ siteleri (www, daire başkanlıkları, fakülteler, bölümler ...) tek bir içerik sisteminden beslenen Next.js
uygulamaları: ham HTML'de metin yok, içerik api.kocaeli.edu.tr'den gelir. Her alt sitenin kimliği
(Admin/Isp/GetIspCodeByName) "code" başlığıyla gönderilir:
  - Menu/GetAll           -> menü -> alt menü -> sayfa bağlantıları ("/sayfalar/<slug>" ya da başka alt site)
  - Page/GetBySlug?Slug=  -> {"page": {"title", "content" (HTML)}, "files": [...], ...}
Gezinme: www'den başlanır; menülerde ve sayfa içeriklerinde geçen her *.kocaeli.edu.tr alt sitesi keşfedilir.

Kayıt: mevzuat/işlem anahtar kelimesi geçen ve yeterince uzun sayfalar ->
  veri/web/sayfalar/<alt_site>__<slug>.txt (kaynak adresi + erişim tarihi başta),
  veri/web/sayfalar.json (dizin), veri/web/pdf_baglantilari.tsv (sayfalardaki PDF'ler; indirilmez).
Siteyi yormamak için istekler arasında bekleme var. Sunucu eski TLS yeniden anlaşması kullanıyor:
sertifika doğrulaması açık, yalnızca OP_LEGACY_SERVER_CONNECT.

Kullanım:  python -X utf8 web_tara.py
"""

from __future__ import annotations

import json
import re
import ssl
import sys
import time
from collections import deque
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "veri" / "web"
PAGES_DIR = OUT_DIR / "sayfalar"
API = "https://api.kocaeli.edu.tr/api"
DELAY = 0.3  # saniye, istekler arası
MAX_PAGES_PER_HOST = 120
MIN_CHARS = 400
START_HOSTS = ["www.kocaeli.edu.tr"]
SKIP_HOSTS = {"api", "avesis", "bapsis", "bilsis", "ibs", "kys", "ogr", "ects", "obs", "hastane", "eduroam",
              "yabancibasvuru", "sanaltur", "radyoki", "cop31", "merlab", "tanitim", "mail", "webmail", "uzem",
              "ekampus", "kutuphane", "bilgiislem"}
KEEP_RE = re.compile(r"yönetmeli|yönerge|usul ve esas|esasları|başvur|staj|sınav|harç|katkı payı|kayıt|mezun|"
                     r"diploma|ders|burs|izin|görevlendir|ek ders|bap|proje|ihale|satın al|harcırah|bütçe|"
                     r"erasmus|farabi|mevlana|değişim|yatay geçiş|dikey geçiş|çift anadal|yandal|muafiyet|"
                     r"intibak|akademik takvim|sık sorulan|nasıl|işlem|form|belge|ücret|sigorta|atama|kadro|"
                     r"terfi|emekli|disiplin|ödünç|yurt|mediko|komisyon|koşul|şart|süre", re.IGNORECASE)


class LegacyAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context()
        context.options |= getattr(ssl, "OP_LEGACY_SERVER_CONNECT", 0x4)
        kwargs["ssl_context"] = context
        return super().init_poolmanager(*args, **kwargs)


def session() -> requests.Session:
    s = requests.Session()
    s.mount("https://", LegacyAdapter())
    s.headers["User-Agent"] = "Mozilla/5.0 (KOU mevzuat asistani veri toplama)"
    return s


class Api:
    def __init__(self) -> None:
        self.http = session()

    def get(self, path: str, code: str | None = None, **params: Any) -> Any:
        time.sleep(DELAY)
        headers = {"accept": "*/*", **({"code": code} if code else {})}
        response = self.http.get(f"{API}/{path}", params=params, headers=headers, timeout=25)
        response.raise_for_status()
        return response.json().get("data")


def html_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for br in soup.find_all(["br", "p", "li", "tr", "h1", "h2", "h3", "h4", "div"]):
        br.append("\n")
    lines = [" ".join(line.split()) for line in soup.get_text().split("\n")]
    return "\n".join(line for line in lines if line)


def links_in(html: str, base: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html or "", "html.parser")
    return [(urljoin(base, a["href"]), " ".join(a.get_text().split())[:120]) for a in soup.find_all("a", href=True)]


def menu_links(menu: list[dict[str, Any]]) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for item in menu or []:
        if item.get("url"):
            found.append((item["url"], item.get("title", "")))
        for link in item.get("menuLinks") or []:
            if link.get("url"):
                found.append((link["url"], link.get("title", "")))
        for sub in item.get("subMenus") or []:
            for entry in sub.get("subMenuItems") or []:
                if entry.get("url"):
                    found.append((entry["url"], entry.get("title", "")))
    return found


def main() -> int:
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    api = Api()
    hosts = deque(START_HOSTS)
    seen_hosts: set[str] = set(START_HOSTS)
    pdfs: dict[str, tuple[str, str]] = {}
    saved: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    stats = {"site": 0, "cms_disi": 0, "sayfa": 0}

    def note_host(url: str) -> str | None:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if not host.endswith("kocaeli.edu.tr") or host.split(".")[0] in SKIP_HOSTS:
            return None
        if host == "kocaeli.edu.tr":
            host = "www.kocaeli.edu.tr"
        if host not in seen_hosts:
            seen_hosts.add(host)
            hosts.append(host)
        return host

    while hosts:
        host = hosts.popleft()
        try:
            info = api.get("Admin/Isp/GetIspCodeByName", IspWebUrl=host)
        except Exception as exc:
            print(f"  {host}: kimlik alınamadı ({str(exc)[:60]})", flush=True)
            continue
        if not info or not info.get("code"):
            stats["cms_disi"] += 1
            continue
        code = str(info["code"])
        stats["site"] += 1
        try:
            menu = api.get("Menu/GetAll", code, TranslationCode=0)
        except Exception:
            menu = []
        base = f"https://{host}/"
        slugs: deque[str] = deque()
        seen_slugs: set[str] = set()

        def consider(url: str) -> None:
            absolute = urljoin(base, url)
            parsed = urlparse(absolute)
            if absolute.lower().endswith(".pdf"):
                return
            target = note_host(absolute)
            if target == host and parsed.path.startswith("/sayfalar/"):
                slug = parsed.path.split("/sayfalar/")[1].strip("/")
                if slug and slug not in seen_slugs:
                    seen_slugs.add(slug)
                    slugs.append(slug)

        for url, _ in menu_links(menu):
            consider(url)
        fetched = 0
        while slugs and fetched < MAX_PAGES_PER_HOST:
            slug = slugs.popleft()
            fetched += 1
            try:
                data = api.get("Page/GetBySlug", code, TranslationCode=0, Slug=slug)
            except Exception:
                continue
            if not data or not data.get("page"):
                continue
            page = data["page"]
            content = page.get("content") or ""
            url = f"{base}sayfalar/{slug}"
            for link, label in links_in(content, base):
                if link.lower().endswith(".pdf"):
                    pdfs.setdefault(link, (label, url))
                else:
                    consider(link)
            for item in data.get("files") or []:
                path = item.get("filePath") or item.get("url") or item.get("path") or ""
                if path.lower().endswith(".pdf"):
                    pdfs.setdefault(urljoin(base, path), (item.get("title") or item.get("name") or "", url))
            title = " ".join((page.get("title") or "").split())
            text = html_text(content)
            stats["sayfa"] += 1
            fingerprint = re.sub(r"\W+", "", text[:1500]).lower()
            if len(text) < MIN_CHARS or not KEEP_RE.search(title + " " + text[:4000]) or fingerprint in fingerprints:
                continue
            fingerprints.add(fingerprint)
            name = f"{host.split('.')[0]}__{re.sub(r'[^a-z0-9]+', '-', slug.lower())[:80]}"
            (PAGES_DIR / f"{name}.txt").write_text(
                f"KAYNAK: {url}\nBAŞLIK: {title}\nERİŞİM: {date.today().isoformat()}\n\n{text}\n", encoding="utf-8")
            saved.append({"dosya": f"{name}.txt", "url": url, "site": host, "baslik": title, "uzunluk": len(text)})
        print(f"  {host}: {fetched} sayfa okundu, toplam kayıt {len(saved)} | kuyrukta {len(hosts)} site", flush=True)

    (OUT_DIR / "sayfalar.json").write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
    with (OUT_DIR / "pdf_baglantilari.tsv").open("w", encoding="utf-8") as output:
        output.write("pdf\tbağlantı metni\tbulunduğu sayfa\n")
        for link, (label, page) in sorted(pdfs.items()):
            output.write(f"{link}\t{label}\t{page}\n")
    print(f"\nSite: {stats['site']} (içerik sistemi dışı: {stats['cms_disi']}), okunan sayfa: {stats['sayfa']}, "
          f"kaydedilen: {len(saved)}, PDF bağlantısı: {len(pdfs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
