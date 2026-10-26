#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hoca ve öğrencileri ilgilendiren ulusal mevzuatı mevzuat.gov.tr'den indirir ve kataloğa ekler.

Arama: /Anasayfa/MevzuatDatatable (sitenin kendi DataTables isteği, başlıkta arama). PDF: /MevzuatMetin/<tur>.<tertip>.<no>.pdf
(güncel, işlenmiş metin). Elimizde olan (2547, 4734, 6245, YÖK lisansüstü/geçiş ...) tekrar alınmaz.
Çıktı: veri/mevzuat/*.pdf + veri/web/katalog_mevzuat.json (belge_isle.py tüm katalog_*.json'ları okur).

Kullanım:  python -X utf8 mevzuat_tara.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "veri" / "mevzuat"
CATALOG = ROOT / "veri" / "web" / "katalog_mevzuat.json"
SITE = "https://www.mevzuat.gov.tr"

# (arama ifadesi, form türü, kısa ad, belge türü, hedef kitle, en fazla kaç sonuç)
WANTED = [
    ("Devlet Memurları Kanunu", "Kanun", "657 sayılı Devlet Memurları Kanunu", "kanun", "personel", 1),
    ("Yükseköğretim Personel Kanunu", "Kanun", "2914 sayılı Yükseköğretim Personel Kanunu", "kanun", "personel", 1),
    ("Sosyal Sigortalar ve Genel Sağlık Sigortası Kanunu", "Kanun", "5510 sayılı Sosyal Sigortalar ve Genel Sağlık Sigortası Kanunu", "kanun", "herkes", 1),
    ("Mesleki Eğitim Kanunu", "Kanun", "3308 sayılı Mesleki Eğitim Kanunu", "kanun", "herkes", 1),
    ("Kamu Mali Yönetimi ve Kontrol Kanunu", "Kanun", "5018 sayılı Kamu Mali Yönetimi ve Kontrol Kanunu", "kanun", "personel", 1),
    ("Kamu İhale Sözleşmeleri Kanunu", "Kanun", "4735 sayılı Kamu İhale Sözleşmeleri Kanunu", "kanun", "personel", 1),
    ("Yükseköğretim Kurumları Teşkilatı Kanunu", "Kanun", "2809 sayılı Yükseköğretim Kurumları Teşkilatı Kanunu", "kanun", "herkes", 1),
    ("Kredi ve Yurtlar", "Kanun", "351 sayılı Yüksek Öğrenim Kredi ve Yurtlar Kurumu Kanunu", "kanun", "ogrenci", 1),
    ("Bilgi Edinme Hakkı Kanunu", "Kanun", "4982 sayılı Bilgi Edinme Hakkı Kanunu", "kanun", "herkes", 1),
    ("Doçentlik Yönetmeliği", "TumYonetmelik", "Doçentlik Yönetmeliği", "yönetmelik", "personel", 1),
    ("Öğretim Üyeliğine Yükseltilme ve Atanma Yönetmeliği", "TumYonetmelik", "Öğretim Üyeliğine Yükseltilme ve Atanma Yönetmeliği", "yönetmelik", "personel", 1),
    ("Akademik Teşvik Ödeneği Yönetmeliği", "TumYonetmelik", "Akademik Teşvik Ödeneği Yönetmeliği", "yönetmelik", "personel", 1),
    ("Görevlendirmelerde Uyulacak", "TumYonetmelik", "Yükseköğretim Kurumlarında Yurt İçi ve Yurt Dışı Görevlendirme Yönetmeliği", "yönetmelik", "personel", 1),
    ("Merkezi Sınav ve Giriş Sınavlarına", "TumYonetmelik", "Öğretim Elemanı Atamalarında Merkezi Sınav ve Giriş Sınavları Yönetmeliği", "yönetmelik", "personel", 1),
    ("Öğrenci Disiplin", "TumYonetmelik", "Yükseköğretim Kurumları Öğrenci Disiplin Yönetmeliği", "yönetmelik", "ogrenci", 1),
    ("Hastalık Raporları ile Hastalık ve Refakat İznine", "TumYonetmelik", "Devlet Memurlarının Hastalık Raporları ile Hastalık ve Refakat İzni Yönetmeliği", "yönetmelik", "personel", 1),
    ("Kültür ve Spor Dairesi", "TumYonetmelik", "Yükseköğretim Kurumları Sağlık, Kültür ve Spor Dairesi Uygulama Yönetmeliği", "yönetmelik", "herkes", 1),
    ("Kısmi Zamanlı Çalıştır", "TumYonetmelik", "Yükseköğretim Kurumlarında Kısmi Zamanlı Öğrenci Çalıştırma Yönetmeliği", "yönetmelik", "ogrenci", 1),
    ("Engelliler Danışma ve Koordinasyon", "TumYonetmelik", "Yükseköğretim Kurumları Engelliler Danışma ve Koordinasyon Yönetmeliği", "yönetmelik", "herkes", 1),
    ("Tıpta ve Diş Hekimliğinde Uzmanlık Eğitimi Yönetmeliği", "TumYonetmelik", "Tıpta ve Diş Hekimliğinde Uzmanlık Eğitimi Yönetmeliği", "yönetmelik", "herkes", 1),
    ("Kredi ve Burs", "TumYonetmelik", "Yükseköğretim Kredi ve Burs Yönetmeliği", "yönetmelik", "ogrenci", 1),
    ("Döner Sermaye Gelirlerinden", "TumYonetmelik", "Yükseköğretim Döner Sermaye Ek Ödeme Yönetmeliği", "yönetmelik", "personel", 1),
    ("Öğrenci Katkısı", "CumhurbaskaniKararlari", "Öğrenci Katkı Payı ve Öğrenim Ücreti Cumhurbaşkanı Kararı", "karar", "ogrenci", 2),
    ("Etik Davranış İlkeleri", "TumYonetmelik", "Kamu Görevlileri Etik Davranış İlkeleri Yönetmeliği", "yönetmelik", "personel", 1),
    ("Disiplin Kurulları ve Disiplin Amirleri Hakkında", "TumYonetmelik", "Disiplin Kurulları ve Disiplin Amirleri Yönetmeliği", "yönetmelik", "personel", 1),
    ("Yıllık İzin Yönetmeliği", "TumYonetmelik", "Devlet Memurları Yıllık İzin Yönetmeliği", "yönetmelik", "personel", 1),
    # Arama ifadesi başlıkta ardışık geçmeli. Öğrenci disiplin hükümleri 2012'den beri 2547 m.54'te, kısmi zamanlı
    # çalıştırma 2547 m.46'da, memur yıllık izni 657 m.102-104'te (ayrı yönetmelikleri yok / yürürlükten kalktı).
    ("Burs Kredi Verilmesine", "Kanun", "5102 sayılı Yüksek Öğrenim Öğrencilerine Burs Kredi Verilmesine İlişkin Kanun", "kanun", "ogrenci", 1),
    ("Gençlik ve Spor Bakanlığı Kredi", "KurumVeKurulusYonetmeligi", "Gençlik ve Spor Bakanlığı Kredi Yönetmeliği (KYK öğrenim kredisi)", "yönetmelik", "ogrenci", 1),
    ("Gençlik ve Spor Bakanlığı Burs", "KurumVeKurulusYonetmeligi", "Gençlik ve Spor Bakanlığı Burs Yönetmeliği (KYK bursu)", "yönetmelik", "ogrenci", 1),
    ("Yurt Hizmetleri", "KurumVeKurulusYonetmeligi", "Gençlik ve Spor Bakanlığı Yurt Hizmetleri Yönetmeliği (KYK yurtları)", "yönetmelik", "ogrenci", 1),
]
TYPE_LEVEL = {"kanun": 1, "yönetmelik": 2, "karar": 2}
# Başlıkta geçmesi gereken ifade (büyük harf): başka kurumların benzer adlı yönetmelikleri alınmasın
# ("Disiplin Kurulları" araması YSK ve Adli Tıp yönetmeliklerini de getiriyordu).
REQUIRE = {"Disiplin Kurulları ve Disiplin Amirleri Yönetmeliği": "^DİSİPLİN KURULLARI",
           "Yükseköğretim Kurumları Öğrenci Disiplin Yönetmeliği": "YÜKSEKÖĞRETİM",
           "Devlet Memurları Yıllık İzin Yönetmeliği": "MEMUR|DEVLET",
           "Kamu Görevlileri Etik Davranış İlkeleri Yönetmeliği": "KAMU GÖREVLİLERİ",
           "Yükseköğretim Kredi ve Burs Yönetmeliği": "YÜKSEKÖĞRETİM|ÖĞRENCİ",
           "Yükseköğretim Döner Sermaye Ek Ödeme Yönetmeliği": "YÜKSEKÖĞRETİM|ÖĞRETİM ELEMAN",
           "Gençlik ve Spor Bakanlığı Yurt Hizmetleri Yönetmeliği (KYK yurtları)": "YURT HİZMETLERİ YÖNETMELİĞİ"}


def clean(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"
    token = re.search(r'name="antiforgerytoken"[^>]*value="([^"]+)"', s.get(SITE, timeout=60).text).group(1)
    catalog = json.loads(CATALOG.read_text(encoding="utf-8")) if CATALOG.exists() else []
    catalog = [c for c in catalog if "Döner Sermayeli Kuruluşlar" not in c["baslik"]]
    have = {c["kisa"].split(" (")[0] for c in catalog} | {c["kisa"] for c in catalog}
    for term, form, short, kind, audience, limit in WANTED:
        if short in have:  # önceki çalıştırmada bulundu
            continue
        body = {"draw": 1, "start": 0, "length": 10, "columns": [], "order": [],
                "search": {"value": "", "regex": False},
                "parameters": {"MevzuatTur": form, "AranacakIfade": term, "TamCumle": "false",
                               "AranacakYer": "Baslik", "antiforgerytoken": token}}
        rows = []
        for attempt in range(3):  # site art arda isteklerde boş sonuç dönebiliyor: bekleyip tekrar dene
            time.sleep(4 + 4 * attempt)
            rows = s.post(f"{SITE}/Anasayfa/MevzuatDatatable", data=json.dumps(body), timeout=60,
                          headers={"Content-Type": "application/json; charset=utf-8", "X-Requested-With": "XMLHttpRequest",
                                   "antiforgerytoken": token, "RequestVerificationToken": token}).json().get("data") or []
            if rows:
                break
        # En yeni resmi gazete tarihli sonuçlar önce (karar/yönetmelik değişiklikleri birden çok kayıt dönebilir)
        rows.sort(key=lambda r: "".join(reversed((r.get("resmiGazeteTarihi") or "00.00.0000").split("."))), reverse=True)
        taken = 0
        for row in rows:
            title = clean(row.get("mevAdi"))
            if "DEĞİŞİKLİK" in title.upper() and form != "CumhurbaskaniKararlari":
                continue
            if short in REQUIRE and not re.search(REQUIRE[short], title.upper()):
                continue
            tur, tertip, no = row.get("mevzuatTur"), row.get("mevzuatTertip"), row.get("mevzuatNo")
            if not (tur and tertip and no):
                continue
            path = OUT / f"{tur}.{tertip}.{no}.pdf"
            if path.with_suffix(".txt").exists():
                path = path.with_suffix(".txt")
            if not path.exists():
                # PDF yolu türe göre değişiyor (kanun: MevzuatMetin/1.5.657.pdf, yönetmelik: MevzuatMetin/yonetmelik/...):
                # mevzuatın kendi sayfasındaki PDF bağlantısı kullanılır.
                time.sleep(2)
                page = s.get(f"{SITE}/{row.get('url')}", timeout=60).text
                links = re.findall(r'href="([^"]+\.pdf)"', page) or [f"{SITE}/MevzuatMetin/{tur}.{tertip}.{no}.pdf"]
                time.sleep(2)
                pdf = s.get(links[0] if links[0].startswith("http") else SITE + links[0], timeout=120)
                if pdf.content.startswith(b"%PDF"):
                    path.write_bytes(pdf.content)
                else:  # PDF'i olmayan eski metin: sayfanın gösterdiği metin (iframe) .txt olarak
                    iframe = re.search(r'src="(/anasayfa/MevzuatFihristDetayIframe[^"]+)"', page)
                    text = ""
                    if iframe:
                        time.sleep(2)
                        from bs4 import BeautifulSoup
                        html = s.get(SITE + iframe.group(1).replace("&amp;", "&"), timeout=120).text
                        text = BeautifulSoup(html, "html.parser").get_text("\n")
                    if len(text.strip()) < 500:
                        print(f"  metin alınamadı: {title[:60]}", flush=True)
                        continue
                    path = path.with_suffix(".txt")
                    path.write_text(text, encoding="utf-8")
            name = short if limit == 1 else f"{short} ({row.get('resmiGazeteTarihi')})"
            catalog.append({"ad": f"mevzuat_{tur}_{tertip}_{no}", "kisa": name, "baslik": title, "tur": kind,
                            "hiyerarsi": TYPE_LEVEL[kind], "yapi": "", "hedef_kitle": audience,
                            "dosya": f"mevzuat/{path.name}", "birim": "tum", "kurum": "TBMM" if kind == "kanun" else "Resmî Gazete",
                            "kapsam": "web", "durum": "guncel", "madde_hedef_kitle": {}, "yururluk_tarihi": "",
                            "yururluk_bitis": "", "kaynak_url": f"{SITE}/{row.get('url')}"})
            print(f"  {name} <- {title[:80]} ({path.stat().st_size // 1024} KB)", flush=True)
            taken += 1
            if taken >= limit:
                break
        if not taken:
            print(f"  BULUNAMADI: {term}", flush=True)
    CATALOG.write_text(json.dumps(catalog, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Mevzuat: {len(catalog)} belge -> {CATALOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
