#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KOÜ web sitelerinin TAM taraması (kullanıcı isteği: "bir eksik bile kalmasın").

web_tara.py'den farkları: site başına sayfa sınırı yok, konu süzgeci yok (tanıtım, program, anabilim dalı
sayfaları da alınır), duyurular (2024 Eylül sonrası) alınır, içerik sistemi dışındaki eski siteler düz HTML olarak
gezilir. Kişi listeleri (akademik/idari kadro, telefon rehberi) alınmaz.

İçerik sistemi (Next.js) siteleri: api.kocaeli.edu.tr, "code" başlığı (web_tara.Api):
  Menu/GetAll, Page/GetBySlug, Announcement/GetAll
Çıktı (veri/web_tam/):
  sayfalar/<site>__<slug>.txt, duyurular/<site>__<tarih>_<n>.txt, html/<site>__<yol>.txt
  dosya_baglantilari.tsv (PDF/Word; indirilmez), siteler.json (site başına sayılar)
Yeniden çalıştırmada var olan dosyalar yeniden yazılır (tarama baştan yapılır).

Kullanım:  python -X utf8 web_tara_tam.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import deque
from datetime import date
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

from bs4 import BeautifulSoup

from web_tara import SKIP_HOSTS, Api, html_text, links_in, menu_links, session

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "veri" / "web_tam"
MAX_PAGES_PER_SITE = 1500
MIN_CHARS = 150
ANNOUNCE_SINCE = "2024-09-01"
PEOPLE_TITLE_RE = re.compile(r"akademik kadro|idari kadro|akademik personel|idari personel|personelimiz|"
                             r"öğretim elemanları|telefon rehberi|iletişim bilgileri|^iletişim$|^kadro", re.IGNORECASE)
PEOPLE_WORD_RE = re.compile(r"\b(Prof\.|Doç\.|Dr\.|Öğr\.|Arş\.|Gör\.|Uzm\.|Okt\.)")
FILE_RE = re.compile(r"\.(pdf|docx?|xlsx?)($|\?)|FileOperation", re.IGNORECASE)
SKIP_PATH_RE = re.compile(r"/(haberler|haber|galeri|foto|video|etkinlikler|en)/", re.IGNORECASE)
EXTRA_SKIP = {"www2", "obs", "uzem", "ekampus", "webmail", "mail", "kys", "avesis", "bapsis", "bilsis", "ibs", "ects",
              "ogr", "hastane", "eduroam", "sanaltur", "tanitim", "yabancibasvuru", "api"}


def is_people_list(title: str, text: str) -> bool:
    return bool(PEOPLE_TITLE_RE.search(title)) or len(PEOPLE_WORD_RE.findall(text)) > max(4, len(text) / 250)


def save(folder: str, name: str, url: str, title: str, text: str, extra: str = "") -> None:
    path = OUT / folder / f"{name[:150]}.txt"
    path.write_text(f"KAYNAK: {url}\nBAŞLIK: {title}\nERİŞİM: {date.today().isoformat()}\n{extra}\n{text}\n",
                    encoding="utf-8")


def crawl_cms(api: Api, host: str, code: str, files: dict, found_hosts: set[str], stats: dict) -> None:
    prefix = host.split(".")[0]
    base = f"https://{host}/"
    slugs: deque[str] = deque()
    seen: set[str] = set()

    def consider(url: str, label: str = "") -> None:
        absolute = urljoin(base, url or "")
        parsed = urlparse(absolute)
        if FILE_RE.search(absolute):
            files.setdefault(absolute, (label, base))
            return
        target = parsed.netloc.lower()
        if target.endswith("kocaeli.edu.tr") and target != host:
            found_hosts.add("www.kocaeli.edu.tr" if target == "kocaeli.edu.tr" else target)
            return
        if target == host and parsed.path.startswith("/sayfalar/"):
            slug = parsed.path.split("/sayfalar/")[1].strip("/")
            if slug and slug not in seen:
                seen.add(slug)
                slugs.append(slug)

    for url, label in menu_links(api.get("Menu/GetAll", code, TranslationCode=0) or []):
        consider(url, label)
    pages = people = 0
    while slugs and pages < MAX_PAGES_PER_SITE:
        slug = slugs.popleft()
        try:
            data = api.get("Page/GetBySlug", code, TranslationCode=0, Slug=slug)
        except Exception:
            continue
        page = (data or {}).get("page")
        if not page:
            continue
        content = page.get("content") or ""
        for link, label in links_in(content, base):
            consider(link, label)
        for item in (data or {}).get("files") or []:
            path = item.get("url") or item.get("filePath") or ""
            if path:
                files.setdefault(f"https://api.kocaeli.edu.tr/api/FileOperation?filePath={path}",
                                 (item.get("title") or item.get("name") or "", f"{base}sayfalar/{slug}"))
        title = " ".join((page.get("title") or "").split())
        text = html_text(content)
        if len(text) < MIN_CHARS:
            continue
        if is_people_list(title, text):
            people += 1
            continue
        pages += 1
        save("sayfalar", f"{prefix}__{re.sub(r'[^a-z0-9]+', '-', slug.lower())}", f"{base}sayfalar/{slug}", title, text)
    # Duyurular
    announcements = 0
    try:
        items = api.get("Announcement/GetAll", code, TranslationCode=0) or []
    except Exception:
        items = []
    for index, item in enumerate(items):
        ann = item.get("announcement") or {}
        start = (ann.get("startDate") or "")[:10]
        if start < ANNOUNCE_SINCE:
            continue
        text = html_text(ann.get("content") or "")
        for f in item.get("files") or []:
            if f.get("url"):
                files.setdefault(f"https://api.kocaeli.edu.tr/api/FileOperation?filePath={f['url']}",
                                 (ann.get("title") or "", f"{base}duyurular"))
        if len(text) < 60:
            continue
        announcements += 1
        save("duyurular", f"{prefix}__{start}_{index}", f"{base}duyurular", " ".join((ann.get("title") or "").split()),
             text, extra=f"YAYIN: {start} – {(ann.get('endDate') or '')[:10]} | {ann.get('contact') or ''}\n")
    stats[host] = {"sayfa": pages, "kisi_listesi_atlanan": people, "duyuru": announcements}


def crawl_html(http, host: str, files: dict, found_hosts: set[str], stats: dict) -> None:
    """İçerik sistemi dışındaki eski siteler (PHP vb.): aynı site içinde gezinip metni alır."""
    prefix = host.split(".")[0]
    queue = deque([f"http://{host}/"])
    seen: set[str] = set()
    pages = 0
    while queue and len(seen) < MAX_PAGES_PER_SITE:
        url = queue.popleft()
        if url in seen:
            continue
        seen.add(url)
        try:
            time.sleep(0.3)
            r = http.get(url, timeout=25)
        except Exception:
            continue
        if r.status_code != 200 or "text/html" not in r.headers.get("Content-Type", ""):
            continue
        r.encoding = r.apparent_encoding or r.encoding
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            link = urldefrag(urljoin(url, a["href"]))[0]
            parsed = urlparse(link)
            if FILE_RE.search(link):
                files.setdefault(link, (" ".join(a.get_text().split())[:120], url))
            elif parsed.netloc.lower() == host and parsed.scheme in ("http", "https") \
                    and not SKIP_PATH_RE.search(parsed.path) and not re.search(r"\.(jpe?g|png|gif|zip|rar)$", parsed.path, re.I):
                queue.append(link)
            elif parsed.netloc.lower().endswith("kocaeli.edu.tr") and parsed.netloc.lower() != host:
                found_hosts.add(parsed.netloc.lower())
        for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
            tag.decompose()
        title = " ".join((soup.title.get_text() if soup.title else "").split())
        text = "\n".join(l for l in (" ".join(x.split()) for x in soup.get_text("\n").split("\n")) if l)
        if len(text) < MIN_CHARS or is_people_list(title, text):
            continue
        pages += 1
        path = re.sub(r"[^a-z0-9]+", "-", (urlparse(url).path + "-" + urlparse(url).query).lower()).strip("-") or "anasayfa"
        save("html", f"{prefix}__{path}", url, title, text)
    stats[host] = {"sayfa": pages, "eski_site": True}


def main() -> int:
    for folder in ("sayfalar", "duyurular", "html"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    api = Api()
    http = session()
    known = json.loads((ROOT / "veri" / "web" / "site_bilgisi.json").read_text(encoding="utf-8"))
    queue = deque(sorted(set(known) | {"www.kocaeli.edu.tr"}))
    done: set[str] = set()
    files: dict[str, tuple[str, str]] = {}
    stats: dict[str, dict] = {}
    while queue:
        host = queue.popleft()
        if host in done or host.split(".")[0] in SKIP_HOSTS | EXTRA_SKIP:
            continue
        done.add(host)
        found: set[str] = set()
        try:
            info = api.get("Admin/Isp/GetIspCodeByName", IspWebUrl=host) or {}
        except Exception:
            info = {}
        try:
            if info.get("code"):
                crawl_cms(api, host, str(info["code"]), files, found, stats)
            else:
                crawl_html(http, host, files, found, stats)
        except Exception as exc:
            print(f"  {host}: HATA {str(exc)[:80]}", flush=True)
            continue
        queue.extend(sorted(found - done))
        print(f"  {host}: {stats.get(host)} | kuyruk {len(queue)}", flush=True)
    with (OUT / "dosya_baglantilari.tsv").open("w", encoding="utf-8") as output:
        output.write("dosya\tbağlantı metni\tbulunduğu sayfa\n")
        output.writelines(f"{url}\t{label}\t{page}\n" for url, (label, page) in sorted(files.items()))
    (OUT / "siteler.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    total = {k: sum(v.get(k, 0) for v in stats.values()) for k in ("sayfa", "duyuru", "kisi_listesi_atlanan")}
    print(f"\nSite: {len(stats)}, {total}, dosya bağlantısı: {len(files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
