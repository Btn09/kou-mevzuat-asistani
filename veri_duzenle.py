#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""yeni_pdfler/ klasöründeki PDF'leri İÇERİKLERİNE göre sınıflandırıp veri/ klasörüne düzenler.

- Belgeler dosya adından değil, PDF'in başındaki başlık metninden tanınır (imza). Dosya adları
  rastgele olabiliyor (OpenFile12.pdf, 7.5.13948.pdf ...).
- Aynı belgenin birden fazla kopyası varsa: metni bozuk olmayan, tarihi en yeni olan seçilir.
- Kullanılacak belgeler anlamlı adlarla veri/<kapsam>/ altına KOPYALANIR; yeni_pdfler/ olduğu gibi kalır.
- veri/belgeler.json: başlık, tür, kurum, kapsam, sürüm, durum, notlar; dışarıda kalanların gerekçesi.
  Sonraki adımlar (chunk'lama, veri üretimi) bu katalogu okur.

Kullanım:
    python -X utf8 veri_duzenle.py
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import unicodedata
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "yeni_pdfler"
EXTRA_SOURCE_DIR = ROOT / "new_veri"  # 2026-10-16: fakülte/MYO alt klasörleri (staj, İME, bitirme, Erasmus)
TARGET_DIR = ROOT / "veri"
CATALOG = TARGET_DIR / "belgeler.json"
TITLE_ZONE_CHARS = 700  # İmza sadece başlık bölgesinde aranır; dayanak maddesindeki atıflar karışmasın.

# Bazı PDF fontları harfleri bozuk veriyor; okurken düzeltilir (chunk'lama da bunu kullanmalı).
TEXT_FIXES = {"൴": "i"}

# var: başlık bölgesinde hepsi geçmeli; yok: hiçbiri geçmemeli (normalize edilmiş, küçük harf, aksansız).
# kisa: cevaplardaki "Kaynak:" satırında görünen ad.
# kapsam: onlisans_lisans | fakulteye_ozel | lisansustu | genel (herkes) | akademik (personel)  (klasör adı)
# durum : guncel | kontrol_et | donemlik (her yıl yenilenmeli) | yururlukten_kalkti (bağlamda uyarı notuyla)
DOCUMENTS = [
    # ---------------- Fakülte / MYO / bölüm belgeleri (new_veri/, 2026-10-16) ----------------
    # Listenin BAŞINDA: dayanak maddelerinde Ön Lisans-Lisans Yönetmeliği (30913) ve Uygulamalı Eğitimler
    # Yönetmeliği başlık bölgesine düşüyor; genel belgelerin imzası bunları yanlışlıkla sahiplenmesin.
    # birim: fakülte/MYO kısaltması (rag_v2 BIRIMLER). "yapi": "serbest" = MADDE numarası olmayan metin.
    {"ad": "kou_muhendislik_fakultesi_staj_yonergesi", "kisa": "KOÜ Mühendislik Fakültesi Staj Yönergesi",
     "var": ["muhendislik fakultesi staj yonergesi"], "baslik": "KOÜ Mühendislik Fakültesi Staj Yönergesi",
     "tur": "yönerge", "kurum": "KOÜ", "kapsam": "fakulteye_ozel", "birim": "muhendislik", "surum": "",
     "durum": "guncel",
     # PDF'te 12. maddenin (Öğrencinin Sorumlulukları) metni yanlışlıkla "MADDE 10-" etiketiyle başlıyor.
     "metin_duzeltmeleri": [["MADDE 10- (1) Staj yapan öğrenciler", "(1) Staj yapan öğrenciler"]]},
    {"ad": "kou_muhendislik_fakultesi_sinav_kurallari", "kisa": "KOÜ Mühendislik Fakültesi Sınav Kuralları",
     "var": ["sinavlarda uyulmasi gereken kurallar"], "baslik": "KOÜ Mühendislik Fakültesi Sınavlarda Uyulması Gereken Kurallar",
     "tur": "duyuru", "kurum": "KOÜ", "kapsam": "fakulteye_ozel", "birim": "muhendislik", "surum": "",
     "durum": "guncel", "yapi": "serbest"},
    {"ad": "kou_teknoloji_fakultesi_ime_yonergesi", "kisa": "KOÜ Teknoloji Fakültesi İşletmede Mesleki Eğitim Yönergesi",
     "var": ["teknoloji fakultesi", "isletmede mesleki egitim yonergesi"],
     "baslik": "KOÜ Teknoloji Fakültesi İşletmede Mesleki Eğitim Yönergesi", "tur": "yönerge", "kurum": "KOÜ",
     "kapsam": "fakulteye_ozel", "birim": "teknoloji", "surum": "", "durum": "guncel"},
    {"ad": "kou_turizm_fakultesi_ime_yonergesi", "kisa": "KOÜ Turizm Fakültesi İşletmede Mesleki Eğitim Yönergesi",
     "var": ["turizm fakultesi", "isletmede mesleki egitim"], "yok": ["meslek yuksekokulu"],
     "baslik": "KOÜ Turizm Fakültesi İşletmede Mesleki Eğitim (İME) Yönergesi", "tur": "yönerge", "kurum": "KOÜ",
     "kapsam": "fakulteye_ozel", "birim": "turizm", "surum": "", "durum": "guncel"},
    {"ad": "kou_ziraat_fakultesi_staj_yonergesi", "kisa": "KOÜ Ziraat Fakültesi Staj Yönergesi",
     "var": ["ziraat fakultesi staj yonergesi"], "baslik": "KOÜ Ziraat Fakültesi Staj Yönergesi", "tur": "yönerge",
     "kurum": "KOÜ", "kapsam": "fakulteye_ozel", "birim": "ziraat", "surum": "", "durum": "guncel"},
    {"ad": "kou_mimarlik_tasarim_fakultesi_staj_yonergesi", "kisa": "KOÜ Mimarlık ve Tasarım Fakültesi Staj Yönergesi",
     "var": ["mimarlik ve tasarim fakultesi", "staj yonergesi"],
     "baslik": "KOÜ Mimarlık ve Tasarım Fakültesi Staj Yönergesi", "tur": "yönerge", "kurum": "KOÜ",
     "kapsam": "fakulteye_ozel", "birim": "mimarlik", "surum": "29/01/2024", "durum": "guncel",
     "bolum_bazli_numara": True},  # her bölümde madde numarası 1'den başlıyor
    {"ad": "kou_denizcilik_fakultesi_staj_yonergesi", "kisa": "KOÜ Denizcilik Fakültesi Staj Uygulama Yönergesi",
     "var": ["denizcilik fakultesi", "staj uygulama yonergesi"],
     "baslik": "KOÜ Denizcilik Fakültesi Staj Uygulama Yönergesi", "tur": "yönerge", "kurum": "KOÜ",
     "kapsam": "fakulteye_ozel", "birim": "denizcilik", "surum": "30/03/2020", "durum": "guncel"},
    {"ad": "kou_denizcilik_isletmeleri_bitirme_projesi_yonergesi",
     "kisa": "KOÜ Denizcilik İşletmeleri Yönetimi Bölümü Bitirme Projesi Yönergesi",
     "var": ["denizcilik isletmeleri yonetimi bolumu", "bitirme projesi yonergesi"],
     "baslik": "KOÜ Denizcilik Fakültesi Denizcilik İşletmeleri Yönetimi Bölümü Bitirme Projesi Yönergesi",
     "tur": "yönerge", "kurum": "KOÜ", "kapsam": "fakulteye_ozel", "birim": "denizcilik", "surum": "",
     "durum": "guncel"},
    {"ad": "kou_rts_bitirme_projesi_uygulama_esaslari",
     "kisa": "KOÜ İletişim Fakültesi RTS Bölümü Bitirme Projesi Uygulama Esasları",
     "var": ["radyo, televizyon ve sinema bolumu", "bitirme projesi uygulama esaslari"],
     "baslik": "KOÜ İletişim Fakültesi Radyo, Televizyon ve Sinema Bölümü Bitirme Projesi Uygulama Esasları",
     "tur": "esaslar", "kurum": "KOÜ", "kapsam": "fakulteye_ozel", "birim": "iletisim", "surum": "Mart 2016",
     "durum": "guncel", "yapi": "serbest"},
    {"ad": "kou_yapay_zeka_makine_ogrenmesi_staj_esaslari",
     "kisa": "KOÜ Yapay Zeka ve Makine Öğrenmesi Bölümü Staj Esasları",
     "var": ["yapay zeka ve makine ogrenmesi bolumu staj esaslari"],
     "baslik": "KOÜ Fen Edebiyat Fakültesi Yapay Zeka ve Makine Öğrenmesi Bölümü Staj Esasları", "tur": "esaslar",
     "kurum": "KOÜ", "kapsam": "fakulteye_ozel", "birim": "fen_edebiyat", "surum": "", "durum": "guncel",
     "yapi": "serbest"},
    {"ad": "kou_yapay_zeka_makine_ogrenmesi_staj_kabul_yazisi",
     "kisa": "KOÜ Yapay Zeka ve Makine Öğrenmesi Bölümü Staj Kabul Yazısı",
     "kaynak_adi": "20260310_61f88", "var": [],
     "baslik": "KOÜ Fen Edebiyat Fakültesi Yapay Zeka ve Makine Öğrenmesi Bölümü Staj Yeri Bilgilendirme Yazısı (İlgili Makama)",
     "tur": "duyuru", "kurum": "KOÜ", "kapsam": "fakulteye_ozel", "birim": "fen_edebiyat", "surum": "10/03/2026",
     "durum": "guncel", "yapi": "serbest",
     "not": "Taranmış tek sayfa (metin katmanı yok); metin elle aktarıldı -> aynı adlı .txt."},
    {"ad": "kou_kocaeli_myo_ime_yonergesi", "kisa": "KOÜ Kocaeli Meslek Yüksekokulu İşletmede Mesleki Eğitim Yönergesi",
     "var": ["kocaeli meslek yuksekokulu", "isletmede mesleki egitim (ime) yonergesi"],
     "baslik": "KOÜ Kocaeli Meslek Yüksekokulu İşletmede Mesleki Eğitim (İME) Yönergesi", "tur": "yönerge",
     "kurum": "KOÜ", "kapsam": "fakulteye_ozel", "birim": "kocaeli_myo", "surum": "", "durum": "guncel",
     "not": "Kocaeli Sağlık Hizmetleri MYO klasöründeki dosya bunun birebir kopyası (o MYO'nun kendi yönergesi değil)."},
    {"ad": "kou_kocaeli_myo_staj_islemleri", "kisa": "KOÜ Kocaeli Meslek Yüksekokulu Staj İşlemleri",
     "var": ["kocaeli meslek yuksekokulu", "staj islemleri"],
     "baslik": "KOÜ Kocaeli Meslek Yüksekokulu Staj İşlemleri", "tur": "duyuru", "kurum": "KOÜ",
     "kapsam": "fakulteye_ozel", "birim": "kocaeli_myo", "surum": "28/05/2024", "durum": "guncel",
     "yapi": "serbest"},
    {"ad": "kou_kartepe_turizm_myo_ime_staj_yonergesi",
     "kisa": "KOÜ Kartepe Turizm MYO İşletmede Mesleki Eğitim ve Staj Yönergesi",
     "var": ["kartepe turizm meslek yuksekokulu", "isletmede mesleki egitim"],
     "baslik": "KOÜ Kartepe Turizm Meslek Yüksekokulu İşletmede Mesleki Eğitim (İME) ve Staj Yönergesi",
     "tur": "yönerge", "kurum": "KOÜ", "kapsam": "fakulteye_ozel", "birim": "kartepe_myo", "surum": "",
     "durum": "guncel"},
    {"ad": "kou_ford_otosan_ihsaniye_myo_staj_ime_yonergesi",
     "kisa": "KOÜ Ford Otosan İhsaniye Otomotiv MYO Staj ve İME Yönergesi",
     "var": ["ford otosan ihsaniye", "isletmede mesleki egitim"],
     "baslik": "KOÜ Ford Otosan İhsaniye Otomotiv Meslek Yüksekokulu Staj ve İşletmede Mesleki Eğitim (İME) Yönergesi",
     "tur": "yönerge", "kurum": "KOÜ", "kapsam": "fakulteye_ozel", "birim": "ford_otosan_myo", "surum": "",
     "durum": "guncel"},
    {"ad": "kou_degirmendere_myo_el_sanatlari_staj_yonergesi",
     "kisa": "KOÜ Değirmendere Ali Özbay MYO El Sanatları Bölümü Staj Yönergesi",
     "var": ["degirmendere ali ozbay", "staj yonergesi"],
     "baslik": "KOÜ Değirmendere Ali Özbay Meslek Yüksekokulu El Sanatları Bölümü Staj Yönergesi", "tur": "yönerge",
     "kurum": "KOÜ", "kapsam": "fakulteye_ozel", "birim": "degirmendere_myo", "surum": "", "durum": "guncel"},
    # ---------------- Erasmus (Uluslararası İlişkiler) ----------------
    {"ad": "kou_ab_proje_hibe_harcama_yonergesi",
     "kisa": "KOÜ AB ve Uluslararası Proje Hibelerinin Harcanması Yönergesi",
     "var": ["proje karsiligi aktarilan hibe tutarlarinin harcanmasi"],
     "baslik": "KOÜ Avrupa Birliği ve Uluslararası Kuruluşların Kaynaklarından Proje Karşılığı Aktarılan Hibe "
               "Tutarlarının Harcanması ve Muhasebeleştirilmesi İşlem Yönergesi",
     "tur": "yönerge", "kurum": "KOÜ", "kapsam": "akademik", "surum": "", "durum": "guncel"},
    {"ad": "kou_erasmus_odeme_kanitlayici_belgeler", "kisa": "KOÜ Erasmus Ödemeleri Kanıtlayıcı Belgeler Listesi",
     "var": ["erasmus kapsaminda yapilacak hibe", "kanitlayici belgeler"],
     "baslik": "Erasmus Kapsamında Yapılacak Hibe ve Diğer Ödemelerde Kanıtlayıcı Belgeler", "tur": "duyuru",
     "kurum": "KOÜ", "kapsam": "akademik", "surum": "", "durum": "guncel", "yapi": "serbest"},
    {"ad": "kou_erasmus_taninma_islemleri_rehberi", "kisa": "KOÜ Erasmus Tanınma İşlemleri Rehberi",
     "var": ["taninma islemleri"], "baslik": "KOÜ Uluslararası İlişkiler Birimi Erasmus Tanınma İşlemleri Rehberi",
     "tur": "rehber", "kurum": "KOÜ", "kapsam": "genel", "surum": "", "durum": "guncel", "yapi": "serbest",
     "not": "Koordinatörler için; ekran görüntüsü sayfaları (5, 9, 12, 14) metinsiz."},
    # ---------------- KOÜ, ön lisans / lisans (tüm öğrenciler) ----------------
    {"ad": "kou_onlisans_lisans_egitim_ogretim_yonetmeligi_2019", "kisa": "KOÜ Ön Lisans ve Lisans Eğitim-Öğretim Yönetmeliği",
     "var": ["on lisans ve lisans egitim-ogretim yonetmeligi", "30913"], "yok": ["degisiklik yapilmasina"],
     "baslik": "KOÜ Ön Lisans ve Lisans Eğitim-Öğretim Yönetmeliği", "tur": "yönetmelik", "kurum": "KOÜ",
     "kapsam": "onlisans_lisans", "surum": "RG 9/10/2019, 30913", "durum": "kontrol_et",
     "not": "2023 değişikliği ayrı belgede (38. ve 39. madde). Sonraki değişiklikler indirilmemiş olabilir; "
            "mevzuat.gov.tr'deki işlenmiş güncel metin tercih edilmeli.",
     "degisiklikler": ["kou_onlisans_lisans_yonetmelik_degisiklik_2023"]},
    {"ad": "kou_onlisans_lisans_yonetmelik_degisiklik_2023", "kisa": "KOÜ Ön Lisans ve Lisans Yönetmeliği Değişikliği (2023)",
     "var": ["on lisans ve lisans egitim-ogretim yonetmeliginde degisiklik", "32313"],
     "baslik": "KOÜ Ön Lisans ve Lisans Eğitim-Öğretim Yönetmeliğinde Değişiklik (iptal kararı işlenmiş)",
     "tur": "değişiklik", "kurum": "KOÜ", "kapsam": "onlisans_lisans",
     "surum": "RG 18/9/2023, 32313; Kocaeli 1. İdare Mah. 22/1/2024 iptal kararı işlenmiş", "durum": "guncel",
     "degistirdigi": "kou_onlisans_lisans_egitim_ogretim_yonetmeligi_2019"},
    {"ad": "kou_basari_notu_degerlendirme_usul_esaslar", "kisa": "KOÜ Başarı Notu Değerlendirme Usul ve Esasları",
     "var": ["basari notunun degerlendirmesine iliskin usul ve esaslar"],
     "baslik": "KOÜ Ön Lisans ve Lisans Öğrencileri İçin Başarı Notunun Değerlendirmesine İlişkin Usul ve Esaslar",
     "tur": "usul ve esaslar", "kurum": "KOÜ", "kapsam": "onlisans_lisans", "surum": "16/07/2019",
     "durum": "guncel"},
    {"ad": "kou_yuzde_on_basari_degerlendirme_yonergesi", "kisa": "KOÜ Yüzde On Başarı Değerlendirme Yönergesi", "var": ["yuzde on basari degerlendirme yonergesi"],
     "baslik": "KOÜ Yüzde On Başarı Değerlendirme Yönergesi", "tur": "yönerge", "kurum": "KOÜ",
     "kapsam": "onlisans_lisans", "surum": "Senato 26/10/2021 değişikliği dahil", "durum": "guncel"},
    {"ad": "kou_cift_anadal_programi_yonetmeligi", "kisa": "KOÜ Çift Anadal Programı Yönetmeliği", "var": ["cift anadal programi yonetmeligi"],
     "baslik": "KOÜ Çift Anadal Programı Yönetmeliği", "tur": "yönetmelik", "kurum": "KOÜ",
     "kapsam": "onlisans_lisans", "surum": "yayım tarihi metinde yok", "durum": "kontrol_et"},
    {"ad": "kou_diploma_diploma_eki_belgeler_yonergesi", "kisa": "KOÜ Diploma ve Belgeler Yönergesi",
     "var": ["diploma, diploma eki ve diger belgelerin duzenlenmesine"],
     "baslik": "KOÜ Diploma, Diploma Eki ve Diğer Belgelerin Düzenlenmesine İlişkin Yönerge", "tur": "yönerge",
     "kurum": "KOÜ", "kapsam": "onlisans_lisans", "surum": "08/02/2022", "durum": "guncel",
     "not": "4. sayfadan sonrası belge örnekleri (resim); metin yalnızca ilk 3 sayfada."},
    {"ad": "kou_ogrenci_danismanlik_hizmetleri_yonergesi", "kisa": "KOÜ Öğrenci Danışmanlık Hizmetleri Yönergesi", "var": ["danismanhk hizmetlerine"],
     "baslik": "KOÜ Öğrenci Danışmanlık Hizmetleri Uygulama Yönergesi", "tur": "yönerge", "kurum": "KOÜ",
     "kapsam": "onlisans_lisans", "surum": "tarih metinde yok", "durum": "kontrol_et",
     "not": "Taranmış PDF, OCR hatalı ('Yönetrneliğini', 'MADDE $-'). Metin düzeltilmeli veya temiz kopya "
            "bulunmalı. AKTS üst limiti burada.",
     "metin_kalitesi": "dusuk_ocr"},
    {"ad": "kou_yabanci_diller_yuksekokulu_egitim_sinav_yonergesi", "kisa": "KOÜ Yabancı Diller Yüksekokulu Yönergesi",
     "var": ["yabanci diller yuksekokulu", "egitim-ogretim ve sinav yonergesi"],
     "baslik": "KOÜ Yabancı Diller Yüksekokulu Eğitim-Öğretim ve Sınav Yönergesi", "tur": "yönerge",
     "kurum": "KOÜ", "kapsam": "onlisans_lisans", "surum": "Eylül 2024 güncellemesi", "durum": "guncel"},
    {"ad": "kou_ozel_ogrenci_yonergesi", "kisa": "KOÜ Özel Öğrenci Yönergesi", "var": ["ozel ogrenci yonergesi"],
     "baslik": "KOÜ Ön Lisans ve Lisans Programları İçin Özel Öğrenci Yönergesi", "tur": "yönerge",
     "kurum": "KOÜ", "kapsam": "onlisans_lisans", "surum": "2021-2022'den itibaren", "durum": "guncel"},
    {"ad": "kou_onceki_ogrenmelerin_taninmasi_yonergesi", "kisa": "KOÜ Önceki Öğrenmelerin Tanınması Yönergesi", "var": ["onceki ogrenmelerin taninmasi yonergesi"],
     "baslik": "KOÜ Önceki Öğrenmelerin Tanınması Yönergesi", "tur": "yönerge", "kurum": "KOÜ",
     "kapsam": "onlisans_lisans", "surum": "09/01/2024", "durum": "guncel"},
    {"ad": "kou_uluslararasi_ogrenci_kabul_yonergesi", "kisa": "KOÜ Uluslararası Öğrenci Kabul Yönergesi", "var": ["uluslararasi ogrenci kabul yonergesi"],
     "baslik": "KOÜ Uluslararası Öğrenci Kabul Yönergesi", "tur": "yönerge", "kurum": "KOÜ",
     "kapsam": "onlisans_lisans", "surum": "tarih metinde yok", "durum": "kontrol_et"},
    {"ad": "kou_butunlesik_yuksek_lisans_uygulama_esaslari", "kisa": "KOÜ Bütünleşik Yüksek Lisans Uygulama Esasları",
     "var": ["butunlesik yuksek lisans programi uygulama esaslari"],
     "baslik": "KOÜ Bütünleşik Yüksek Lisans Programı Uygulama Esasları", "tur": "uygulama esasları",
     "kurum": "KOÜ", "kapsam": "onlisans_lisans", "surum": "08/09/2025", "durum": "guncel",
     "not": "Lisans öğrencilerinin yüksek lisansa erken başlaması."},
    {"ad": "kou_muafiyet_ve_intibak_yonergesi", "kisa": "KOÜ Muafiyet ve İntibak Yönergesi",
     "var": ["muafiyet ve intibak"],
     "baslik": "KOÜ Muafiyet ve İntibak Yönergesi", "tur": "yönerge", "kurum": "KOÜ",
     "kapsam": "onlisans_lisans", "surum": "2019-2020'den itibaren", "durum": "guncel",
     "not": "Taranmış PDF, OCR çok hatalı (başlık bile okunmuyor); sayfa resimlerinden elle düzeltilmiş .txt kullanılır.",
     "metin_kalitesi": "dusuk_ocr"},
    {"ad": "kou_yaz_ogretimi_esaslari", "kisa": "KOÜ Yaz Öğretimine İlişkin Esaslar",
     "var": ["yaz ogretimine iliskin esaslar"],
     "baslik": "KOÜ Yaz Öğretimine İlişkin Esaslar", "tur": "esaslar", "kurum": "KOÜ",
     "kapsam": "onlisans_lisans", "surum": "Senato 27.07.2023 değişikliği dahil", "durum": "guncel",
     "not": "Maddeler 'ESAS 1-' biçiminde numaralı. Diğer üniversitelerin yaz okulundan ders alma."},
    {"ad": "kou_basariya_dayali_yatay_gecis_kilavuzu_2026_2027", "kisa": "KOÜ Başarıya Dayalı Yatay Geçiş Kılavuzu (2026-2027 Güz)",
     "var": ["basariya dayali", "yatay gecis"],
     "baslik": "KOÜ 2026-2027 Güz Yarıyılı Başarıya Dayalı Kurum İçi - Kurumlar Arası Yatay Geçiş Kılavuzu",
     "tur": "kılavuz", "kurum": "KOÜ", "kapsam": "onlisans_lisans", "surum": "2026-2027 güz yarıyılı",
     "durum": "donemlik",
     "not": "Döneme özel (takvim ve kontenjan). Her yıl yenisiyle değiştirilmeli. Madde yapısı yok."},
    # ---------------- Genel (öğrenci + personel) ----------------
    {"ad": "kanun_2547_yuksekogretim", "kisa": "2547 sayılı Yükseköğretim Kanunu",
     "var": ["yuksekogretim kanunu kanun numarasi : 2547"],
     "baslik": "2547 sayılı Yükseköğretim Kanunu (işlenmiş güncel metin)", "tur": "kanun", "kurum": "TBMM",
     "kapsam": "akademik", "surum": "mevzuat.gov.tr işlenmiş metin; 7437 (2023) ve AYM 22/2/2024 iptali dahil",
     "durum": "guncel",
     "not": "Öğrenci disiplin hükümleri (Madde 54) ile öğretim elemanlarının statüsü, görev, izin, atama hükümleri. "
            "7437 sayılı torba kanundan ayrıca alınmaz: bu metin ondan daha günceldir."},
    # ---------------- Akademik ve idari personel ----------------
    {"ad": "kou_ders_gorevlendirme_ek_ders_yonergesi", "kisa": "KOÜ Ders Görevlendirme ve Ek Ders Ücreti Yönergesi",
     "var": ["ders gorevlendirmesi ve ek ders ucreti"],
     "baslik": "KOÜ Ders Görevlendirmesi ve Ek Ders Ücreti Ödemelerine İlişkin Usul ve Esaslar Hakkında Yönerge",
     "tur": "yönerge", "kurum": "KOÜ", "kapsam": "akademik", "surum": "", "durum": "kontrol_et"},
    {"ad": "kou_bap_uygulama_yonergesi", "kisa": "KOÜ BAP Uygulama Yönergesi",
     "var": ["bilimsel arastirma projeleri uygulama yonergesi"],
     "baslik": "KOÜ Bilimsel Araştırma Projeleri Uygulama Yönergesi", "tur": "yönerge", "kurum": "KOÜ",
     "kapsam": "akademik", "surum": "en yeni kopya (2026'ya kadar değişiklikler)", "durum": "guncel"},
    {"ad": "kou_bap_uygulama_esaslari", "kisa": "KOÜ BAP Uygulama Esasları",
     "var": ["bap uygulama esaslari"],
     "baslik": "KOÜ Bilimsel Araştırma Projeleri Koordinasyon Birimi BAP Uygulama Esasları", "tur": "kılavuz",
     "kurum": "KOÜ", "kapsam": "akademik", "surum": "2026", "durum": "guncel",
     "not": "Madde yapısı yok; bölüm başlıklarına göre ayrılır."},
    {"ad": "yok_bap_yonetmeligi", "kisa": "YÖK Bilimsel Araştırma Projeleri Yönetmeliği",
     "var": ["bilimsel arastirma", "projeleri hakkinda yonetmelik"],
     "baslik": "Yükseköğretim Kurumları Bilimsel Araştırma Projeleri Hakkında Yönetmelik", "tur": "yönetmelik",
     "kurum": "YÖK", "kapsam": "akademik", "surum": "16/7/2020 değişikliği dahil", "durum": "guncel"},
    {"ad": "kamu_ihale_3f_yuksekogretim_alimlari", "kisa": "Yükseköğretim Kurumlarının 4734/3(f) Kapsamındaki Alımları Usul ve Esasları",
     "var": ["9652", "4734"],
     "baslik": "Yükseköğretim Kurumları Tarafından 4734 Sayılı Kamu İhale Kanununun 3(f) Bendi Kapsamında "
               "Yapılacak Alımlara İlişkin Usul ve Esaslar",
     "tur": "usul ve esaslar", "kurum": "Cumhurbaşkanlığı", "kapsam": "akademik",
     "surum": "20.03.2025, 9652 sayılı Cumhurbaşkanı Kararı", "durum": "guncel",
     "not": "Taranmış PDF, OCR orta kalitede (bazı harfler bozuk). BAP alımları.",
     # OCR'ın bozduğu madde etiketleri; PDF'teki sırayla tek tek kontrol edildi.
     "metin_duzeltmeleri": [["MADDE G (l)", "MADDE 6- (l)"], ["MADDE l0-", "MADDE 10-"], ["MADDE l1-", "MADDE 11-"],
                            ["MADDE 15_", "MADDE 15-"], ["MADDE t6-", "MADDE 16-"], ["MADDE r8-", "MADDE 18-"],
                            ["LIADDE 2E-", "MADDE 28-"], ["MADDE 4l-", "MADDE 41-"], ["MADDI 46-", "MADDE 46-"],
                            ["MADDE a9-", "MADDE 49-"], ["MADDE 5a-", "MADDE 54-"]]},
    {"ad": "kanun_4734_kamu_ihale", "kisa": "4734 sayılı Kamu İhale Kanunu",
     "var": ["kamu ihale kanunu", "4734"], "yok": ["9652"],
     "baslik": "4734 sayılı Kamu İhale Kanunu", "tur": "kanun", "kurum": "TBMM", "kapsam": "akademik",
     "surum": "mevzuat.gov.tr işlenmiş metin", "durum": "guncel"},
    {"ad": "kanun_6245_harcirah", "kisa": "6245 sayılı Harcırah Kanunu", "var": ["harcirah kanunu"],
     "baslik": "6245 sayılı Harcırah Kanunu", "tur": "kanun", "kurum": "TBMM", "kapsam": "akademik",
     "surum": "mevzuat.gov.tr işlenmiş metin", "durum": "guncel", "not": "Yurt içi/dışı görev yolluğu."},
    {"ad": "gib_kamu_e_fatura_duyurusu", "kisa": "GİB Kamu Kurumlarına Düzenlenen e-Faturalar Duyurusu",
     "var": ["kamu kurumlarina duzenlenen e-faturalar"],
     "baslik": "Gelir İdaresi Başkanlığı – Kamu Kurumlarına Düzenlenen e-Faturalar Hakkında Duyuru",
     "tur": "kılavuz", "kurum": "GİB", "kapsam": "akademik", "surum": "23/02/2021", "durum": "kontrol_et"},
    {"ad": "kdv_genel_tebligi_117_tevkifat", "kisa": "KDV Genel Tebliği (Seri No: 117)",
     "var": ["katma deger vergisi", "seri no: 117"],
     "baslik": "Katma Değer Vergisi Genel Tebliği (Seri No: 117) – KDV Tevkifat Uygulaması",
     "tur": "tebliğ", "kurum": "Maliye Bakanlığı", "kapsam": "akademik", "surum": "RG 14/4/2012, 28264",
     "durum": "yururlukten_kalkti",
     "not": "Kullanıcı isteğiyle eklendi. 1/5/2014'te KDV Genel Uygulama Tebliği ile yürürlükten kalktı; "
            "bağlamda her seferinde bu uyarı gösterilir.",
     # Model bu metni güncel kural gibi sunmasın: her bağlam bloğunda belge adının altında gösterilir.
     "baglam_notu": "Bu tebliğ 1/5/2014 tarihinde yürürlükten kalkmıştır; güncel hükümler KDV Genel Uygulama "
                    "Tebliği'nin 'I/C Tevkifat Uygulaması' bölümündedir. Bilgiyi verirken bunu mutlaka belirt."},
    {"ad": "kou_kalite_yonergesi", "kisa": "KOÜ Kalite Yönergesi", "var": ["kalite yonergesi"],
     "baslik": "KOÜ Kalite Yönergesi", "tur": "yönerge", "kurum": "KOÜ", "kapsam": "akademik", "surum": "",
     "durum": "kontrol_et"},
    {"ad": "kou_egitim_koordinatorlugu_komisyonu_yonergesi", "kisa": "KOÜ Eğitim Koordinatörlüğü ve Eğitim Komisyonu Yönergesi",
     "var": ["egitim koordinatorlugu ve egitim komisyonu"],
     "baslik": "KOÜ Eğitim Koordinatörlüğü ve Eğitim Komisyonu Yönergesi", "tur": "yönerge", "kurum": "KOÜ",
     "kapsam": "akademik", "surum": "04/04/2023", "durum": "guncel"},
    {"ad": "yok_akademik_teskilat_yonetmeligi", "kisa": "Üniversitelerde Akademik Teşkilat Yönetmeliği",
     "var": ["akademik teskilat yonetmeligi"],
     "baslik": "Üniversitelerde Akademik Teşkilat Yönetmeliği", "tur": "yönetmelik", "kurum": "YÖK",
     "kapsam": "akademik", "surum": "mevzuat.gov.tr işlenmiş metin", "durum": "guncel"},
    {"ad": "lisansustu_enstitu_teskilat_isleyis_yonetmeligi", "kisa": "Lisansüstü Eğitim-Öğretim Enstitülerinin Teşkilat ve İşleyiş Yönetmeliği",
     "var": ["enstitulerinin teskilat ve isleyis"],
     "baslik": "Lisansüstü Eğitim-Öğretim Enstitülerinin Teşkilat ve İşleyiş Yönetmeliği", "tur": "yönetmelik",
     "kurum": "YÖK", "kapsam": "akademik", "surum": "mevzuat.gov.tr işlenmiş metin", "durum": "guncel"},
    {"ad": "yok_doktorali_arastirmaci_gorevlendirme_usul_esaslar",
     "kisa": "Doktoralı Nitelikli Araştırmacıların Kısmi Zamanlı Görevlendirilmesi Usul ve Esasları",
     "var": ["nitelikli arastirmacilarin"],
     "baslik": "Doktora Derecesine Sahip Türk veya Yabancı Uyruklu Nitelikli Araştırmacıların Yükseköğretim "
               "Kurumlarında Kısmi Zamanlı Görevlendirilmesine İlişkin Usul ve Esaslar",
     "tur": "usul ve esaslar", "kurum": "YÖK", "kapsam": "akademik", "surum": "23/11/2023", "durum": "guncel"},
    # ---------------- YÖK (ulusal), ön lisans / lisans ----------------
    {"ad": "yok_gecis_cift_anadal_yandal_kredi_transferi_yonetmeligi", "kisa": "YÖK Geçiş, Çift Anadal, Yan Dal ve Kredi Transferi Yönetmeliği",
     "var": ["programlar arasinda gecis, cift anadal, yan dal"],
     "baslik": "YÖK Önlisans ve Lisans Düzeyindeki Programlar Arasında Geçiş, Çift Anadal, Yan Dal ile Kurumlar "
               "Arası Kredi Transferi Yapılması Esaslarına İlişkin Yönetmelik",
     "tur": "yönetmelik", "kurum": "YÖK", "kapsam": "onlisans_lisans", "surum": "mevzuat.gov.tr işlenmiş metin",
     "durum": "guncel", "not": "Yatay geçiş ve yan dal soruları buradan cevaplanır."},
    {"ad": "yok_uygulamali_egitimler_cerceve_yonetmeligi", "kisa": "YÖK Uygulamalı Eğitimler Çerçeve Yönetmeliği", "var": ["uygulamali egitimler cerceve yonetmeligi"],
     "baslik": "YÖK Yükseköğretimde Uygulamalı Eğitimler Çerçeve Yönetmeliği", "tur": "yönetmelik",
     "kurum": "YÖK", "kapsam": "onlisans_lisans", "surum": "mevzuat.gov.tr işlenmiş metin", "durum": "guncel",
     "not": "Staj / işyeri eğitimi."},
    {"ad": "yok_uzaktan_ogretim_usul_esaslar", "kisa": "YÖK Uzaktan Öğretim Usul ve Esasları", "var": ["uzaktan ogretime iliskin usul ve esaslar"],
     "baslik": "YÖK Yükseköğretim Kurumlarında Uzaktan Öğretime İlişkin Usul ve Esaslar",
     "tur": "usul ve esaslar", "kurum": "YÖK", "kapsam": "onlisans_lisans",
     "surum": "14.09.2022 değişikliği dahil", "durum": "guncel"},
    {"ad": "yok_myo_mezunlarinin_lisansa_devami_dikey_gecis_yonetmeligi", "kisa": "YÖK Dikey Geçiş Yönetmeliği",
     "var": ["acikogretim on lisans programlari mezunlarinin lisans"],
     "baslik": "YÖK Meslek Yüksekokulları ve Açıköğretim Ön Lisans Programları Mezunlarının Lisans Öğrenimine "
               "Devamları Hakkında Yönetmelik",
     "tur": "yönetmelik", "kurum": "YÖK", "kapsam": "onlisans_lisans", "surum": "son değişiklik 2016",
     "durum": "kontrol_et", "not": "Eski metin; yürürlükte olup olmadığı doğrulanmalı."},
    {"ad": "yok_lisansi_tamamlamayanlarin_onlisans_diplomasi_yonetmeligi", "kisa": "YÖK Lisansı Tamamlayamayanlara Ön Lisans Diploması Yönetmeliği",
     "var": ["tamamlayamayanlarin on lisans diplomasi"],
     "baslik": "YÖK Lisans Öğrenimlerini Tamamlamayan veya Tamamlayamayanların Ön Lisans Diploması Almaları veya "
               "Meslek Yüksekokullarına İntibakları Hakkında Yönetmelik",
     "tur": "yönetmelik", "kurum": "YÖK", "kapsam": "onlisans_lisans", "surum": "eski metin",
     "durum": "kontrol_et",
     "not": "PDF fontu 'i' harfini bozuk veriyor; okurken TEXT_FIXES ile düzeltilmeli."},
    # ---------------- Fakülteye özel (genel ön lisans/lisans yönetmeliği bunlara uygulanmaz) ----------------
    {"ad": "kou_tip_fakultesi_egitim_ogretim_sinav_yonetmeligi", "kisa": "KOÜ Tıp Fakültesi Eğitim-Öğretim ve Sınav Yönetmeliği",
     "var": ["tip fakultesi egitim-ogretim ve sinav yonetmeligi"],
     "baslik": "KOÜ Tıp Fakültesi Eğitim-Öğretim ve Sınav Yönetmeliği", "tur": "yönetmelik", "kurum": "KOÜ",
     "kapsam": "fakulteye_ozel", "surum": "RG 17/11/2025, 33080", "durum": "guncel"},
    {"ad": "kou_dis_hekimligi_fakultesi_egitim_ogretim_yonetmeligi", "kisa": "KOÜ Diş Hekimliği Fakültesi Eğitim-Öğretim Yönetmeliği",
     "var": ["dis hekimligi fakultesi egitim-ogretim yonetmeligi"],
     "baslik": "KOÜ Diş Hekimliği Fakültesi Eğitim-Öğretim Yönetmeliği", "tur": "yönetmelik", "kurum": "KOÜ",
     "kapsam": "fakulteye_ozel", "surum": "RG 18/9/2022, 31957", "durum": "guncel"},
    {"ad": "kou_devlet_konservatuvari_lisans_egitim_sinav_yonetmeligi", "kisa": "KOÜ Devlet Konservatuvarı Eğitim-Öğretim ve Sınav Yönetmeliği",
     "var": ["devlet konservatuvari lisans egitim-ogretim ve sinav yonetmeligi"], "yok": ["degisiklik yapilmasina"],
     "baslik": "KOÜ Devlet Konservatuvarı Lisans Eğitim-Öğretim ve Sınav Yönetmeliği", "tur": "yönetmelik",
     "kurum": "KOÜ", "kapsam": "fakulteye_ozel", "surum": "RG tarihi metinde yok", "durum": "kontrol_et",
     "not": "2022 değişikliği '13/7/2006 tarihli Önlisans-Lisans' yönetmeliğine atıf yapıyor; bu belgenin o "
            "yönetmelikle aynı mı, daha yeni mi olduğu doğrulanmalı."},
    {"ad": "kou_devlet_konservatuvari_yonetmelik_degisiklik_2022", "kisa": "KOÜ Devlet Konservatuvarı Yönetmeliği Değişikliği (2022)",
     "var": ["devlet konservatuvari onlisans-lisans", "degisiklik yapilmasina"],
     "baslik": "KOÜ Devlet Konservatuvarı Önlisans-Lisans Eğitim-Öğretim ve Sınav Yönetmeliğinde Değişiklik",
     "tur": "değişiklik", "kurum": "KOÜ", "kapsam": "fakulteye_ozel", "surum": "RG 30/10/2022, 31998",
     "durum": "kontrol_et", "degistirdigi": "kou_devlet_konservatuvari_lisans_egitim_sinav_yonetmeligi"},
    # ---------------- Lisansüstü ----------------
    {"ad": "kou_lisansustu_egitim_ogretim_yonetmeligi_2023", "kisa": "KOÜ Lisansüstü Eğitim ve Öğretim Yönetmeliği",
     "var": ["lisansustu egitim ve", "ogretim yonetmeligi", "32313"],
     "yok": ["degisiklik yapilmasina", "uygulama esaslari"],
     "baslik": "KOÜ Lisansüstü Eğitim ve Öğretim Yönetmeliği", "tur": "yönetmelik", "kurum": "KOÜ",
     "kapsam": "lisansustu", "surum": "RG 18/9/2023, 32313", "durum": "guncel",
     "degisiklikler": ["kou_lisansustu_yonetmelik_degisiklik_2024"]},
    {"ad": "kou_lisansustu_yonetmelik_degisiklik_2024", "kisa": "KOÜ Lisansüstü Yönetmeliği Değişikliği (2024)",
     "var": ["lisansustu egitim ve ogretim yonetmeliginde degisiklik", "32615"],
     "baslik": "KOÜ Lisansüstü Eğitim ve Öğretim Yönetmeliğinde Değişiklik", "tur": "değişiklik", "kurum": "KOÜ",
     "kapsam": "lisansustu", "surum": "RG 28/7/2024, 32615", "durum": "guncel",
     "degistirdigi": "kou_lisansustu_egitim_ogretim_yonetmeligi_2023"},
    {"ad": "kou_fen_bilimleri_enstitusu_uygulama_esaslari", "kisa": "KOÜ Fen Bilimleri Enstitüsü Uygulama Esasları", "var": ["fen bilimleri enstitusu uygulama esaslari"],
     "baslik": "KOÜ Fen Bilimleri Enstitüsü Lisansüstü Uygulama Esasları", "tur": "uygulama esasları",
     "kurum": "KOÜ", "kapsam": "lisansustu", "surum": "06.01.2026", "durum": "guncel"},
    {"ad": "kou_lisansustu_uzmanlik_alan_dersleri_uygulama_esaslari", "kisa": "KOÜ Uzmanlık Alan Dersleri Uygulama Esasları",
     "var": ["uzmanlik alan dersleri uygulama esaslari"],
     "baslik": "KOÜ Lisansüstü Uzmanlık Alan Dersleri Uygulama Esasları", "tur": "uygulama esasları",
     "kurum": "KOÜ", "kapsam": "lisansustu", "surum": "12/03/2024", "durum": "guncel"},
    {"ad": "yok_lisansustu_egitim_ogretim_yonetmeligi", "kisa": "YÖK Lisansüstü Eğitim ve Öğretim Yönetmeliği",
     "var": ["lisansustu egitim ve ogretim yonetmeligi", "yuksek lisans, doktora ve"], "yok": ["kocaeli"],
     "baslik": "YÖK Lisansüstü Eğitim ve Öğretim Yönetmeliği", "tur": "yönetmelik", "kurum": "YÖK",
     "kapsam": "lisansustu", "surum": "mevzuat.gov.tr işlenmiş metin", "durum": "guncel"},
    {"ad": "yok_lisansustu_tezlerin_elektronik_ortamda_yonergesi", "kisa": "YÖK Lisansüstü Tezlerin Elektronik Ortamda Yayımlanması Yönergesi",
     "var": ["tezlerin elektronik ortamda toplanmasi"],
     "baslik": "YÖK Lisansüstü Tezlerin Elektronik Ortamda Toplanması, Düzenlenmesi ve Erişime Açılmasına "
               "İlişkin Yönerge",
     "tur": "yönerge", "kurum": "YÖK", "kapsam": "lisansustu", "surum": "", "durum": "kontrol_et"},
]

# Dışarıda bırakılacak belgelerin imzaları ve gerekçeleri.
EXCLUDED = [
    (["lisansustu egitim ve ogretim yonetmeligi", "30544"], "Eski sürüm (2018); yerine 2023 yönetmeliği kullanılıyor."),
    (["sosyal bilimler enstitusu uygulama esaslari onerisi"],
     "Taslak ('önerisi'); yürürlükteki kural olduğu belli değil."),
    (["7437", "2547 sayili yuksekogretim kanununun 54"],
     "7437 sayılı torba kanun: öğrenci disiplin hükmü (2547/54), 2547'nin işlenmiş güncel metninde AYM 2024 "
     "iptaliyle birlikte zaten var; diğer maddeleri memur/teşkilat değişiklikleri."),
]


# ---------------- Parça şemasında kullanılan belge düzeyi alanlar ----------------
# Çelişkide üst norm önce: kanun > yönetmelik > tebliğ/yönerge > esaslar > kılavuz/duyuru.
HIYERARSI = {"kanun": 1, "yönetmelik": 2, "tebliğ": 3, "yönerge": 3, "usul ve esaslar": 4,
             "uygulama esasları": 4, "esaslar": 4, "kılavuz": 5, "rehber": 5, "duyuru": 5, "değişiklik": 2}
# Klasör (kapsam) -> (hedef_kitle, birim). Fakülteye özel belgelerin birimi aşağıda belge bazında.
KAPSAM_VARSAYILAN = {"onlisans_lisans": ("ogrenci", "tum"), "fakulteye_ozel": ("ogrenci", None),
                     "lisansustu": ("ogrenci", "enstitu"), "akademik": ("personel", "tum"),
                     "genel": ("personel", "tum")}
BIRIM = {"kou_tip_fakultesi_egitim_ogretim_sinav_yonetmeligi": "tip",
         "kou_dis_hekimligi_fakultesi_egitim_ogretim_yonetmeligi": "dis",
         "kou_devlet_konservatuvari_lisans_egitim_sinav_yonetmeligi": "konservatuvar",
         "kou_devlet_konservatuvari_yonetmelik_degisiklik_2022": "konservatuvar"}
# Hem öğrenciyi hem personeli doğrudan ilgilendirenler.
HEDEF_KITLE = {"kou_ogrenci_danismanlik_hizmetleri_yonergesi": "herkes",
               "yok_uzaktan_ogretim_usul_esaslar": "herkes",
               "kou_erasmus_taninma_islemleri_rehberi": "herkes"}
# Madde bazında farklı kitle (2547: öğrenim süresi, katkı payı, öğrenci disiplini).
MADDE_HEDEF_KITLE = {"kanun_2547_yuksekogretim": {"MADDE 44": "ogrenci", "MADDE 46": "ogrenci",
                                                  "MADDE 54": "ogrenci"}}
# (yürürlük, bitiş) YYYY-AA-GG. Bilinmeyenler boş; belge kimlikleri ayrı veritabanından gelince doldurulur.
YURURLUK = {
    "kou_onlisans_lisans_egitim_ogretim_yonetmeligi_2019": ("2019-10-09", ""),
    "kou_lisansustu_egitim_ogretim_yonetmeligi_2023": ("2023-09-18", ""),
    "kou_tip_fakultesi_egitim_ogretim_sinav_yonetmeligi": ("2025-11-17", ""),
    "kou_dis_hekimligi_fakultesi_egitim_ogretim_yonetmeligi": ("2022-09-18", ""),
    "kanun_2547_yuksekogretim": ("1981-11-06", ""),
    "kdv_genel_tebligi_117_tevkifat": ("2012-05-01", "2014-05-01"),
    # Döneme özel: 2026-2027 güz yarıyılı (başvuru takvimi 03.08.2026'da başlıyor).
    "kou_basariya_dayali_yatay_gecis_kilavuzu_2026_2027": ("2026-08-03", "2027-02-01"),
}


def schema_fields(spec: dict) -> dict:
    hedef_kitle, birim = KAPSAM_VARSAYILAN[spec["kapsam"]]
    yururluk, bitis = YURURLUK.get(spec["ad"], ("", ""))
    return {
        "hiyerarsi": HIYERARSI[spec["tur"]],
        "hedef_kitle": HEDEF_KITLE.get(spec["ad"], hedef_kitle),
        "madde_hedef_kitle": MADDE_HEDEF_KITLE.get(spec["ad"], {}),
        "birim": spec.get("birim") or BIRIM.get(spec["ad"], birim),
        "yururluk_tarihi": yururluk,
        "yururluk_bitis": bitis,
    }


def normalize(text: str) -> str:
    for bad, good in TEXT_FIXES.items():
        text = text.replace(bad, good)
    text = text.casefold().replace("ı", "i")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text)


def read_pdf(path: Path) -> dict:
    with pymupdf.open(path) as pdf:
        text = "\n".join(page.get_text("text", sort=True) for page in pdf)
    norm = normalize(text)
    letters = [c for c in norm if c.isalpha()]
    years = [int(y) for y in re.findall(r"\b\d{1,2}[./]\d{1,2}[./]((?:19|20)\d{2})\b", text)]
    return {
        "path": path,
        "md5": hashlib.md5(path.read_bytes()).hexdigest()[:10],
        "title_zone": norm[:TITLE_ZONE_CHARS],
        # Türkçe metinde 'i' harfi ~%8; fontu bozuk PDF'lerde (i'ler eksik) belirgin şekilde düşer.
        "i_ratio": norm.count("i") / max(1, len(letters)),
        "latest_year": max(years, default=0),
    }


def matches(doc: dict, var: list[str], yok: list[str] | None = None) -> bool:
    zone = doc["title_zone"]
    return all(term in zone for term in var) and not any(term in zone for term in (yok or []))


def main() -> int:
    paths = sorted(SOURCE_DIR.glob("*.pdf")) + sorted(EXTRA_SOURCE_DIR.rglob("*.pdf"))
    docs = [read_pdf(path) for path in paths]
    # Eski çalıştırmadan kalan PDF kopyaları karışmasın. Sadece PDF'ler silinir: yanlarındaki elle
    # düzeltilmiş .txt metinler (ör. taranmış belgelerin temiz metni) korunmalı.
    for folder in {TARGET_DIR / spec["kapsam"] for spec in DOCUMENTS}:
        for stale in folder.glob("*.pdf"):
            stale.unlink()
    catalog: dict[str, list] = {"kullanilan": [], "disarida_birakilan": []}
    assigned: set[Path] = set()

    for spec in DOCUMENTS:
        # Bir dosya tek bir belgeye atanır (ör. ÇAP yönetmeliği YÖK geçiş yönetmeliğinin adını anıyor).
        # Taranmış (metinsiz) belgeler başlıktan tanınamaz: dosya adındaki benzersiz parçayla eşleşir.
        candidates = [d for d in docs if d["path"] not in assigned and (
            spec["kaynak_adi"] in d["path"].name if spec.get("kaynak_adi") else matches(d, spec["var"], spec.get("yok")))]
        if not candidates:
            print(f"UYARI: bulunamadı: {spec['baslik']}")
            continue
        # Önce metni sağlam olan (i oranı normal), sonra en yeni tarihli kopya.
        best = max(candidates, key=lambda d: (d["i_ratio"] > 0.05, d["latest_year"], d["i_ratio"]))
        target = TARGET_DIR / spec["kapsam"] / f"{spec['ad']}.pdf"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best["path"], target)
        entry = {k: v for k, v in spec.items() if k not in ("var", "yok", "kaynak_adi")}
        entry.update({"dosya": target.relative_to(TARGET_DIR).as_posix(), "kaynak_dosya": best["path"].name})
        entry.update(schema_fields(spec))
        catalog["kullanilan"].append(entry)
        assigned.add(best["path"])
        for other in candidates:
            if other is best:
                continue
            assigned.add(other["path"])
            if other["md5"] == best["md5"]:
                reason = "Birebir kopya."
            elif other["i_ratio"] <= 0.05 or other["latest_year"] == best["latest_year"]:
                reason = "Aynı belgenin başka bir kopyası; seçilen kopyanın metni daha temiz okunuyor."
            else:
                reason = "Aynı belgenin eski sürümü."
            catalog["disarida_birakilan"].append(
                {"kaynak_dosya": other["path"].name, "belge": spec["baslik"], "gerekce": reason})

    for doc in docs:
        if doc["path"] in assigned:
            continue
        reason = next((why for var, why in EXCLUDED if matches(doc, var)), None)
        if reason is None:
            print(f"UYARI: sınıflandırılmamış: {doc['path'].name} -> {doc['title_zone'][:120]}")
            reason = "SINIFLANDIRILMADI — elle bakılmalı."
        catalog["disarida_birakilan"].append(
            {"kaynak_dosya": doc["path"].name, "belge": doc["title_zone"][:110].strip(), "gerekce": reason})

    CATALOG.write_text(json.dumps(catalog, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Kullanılan: {len(catalog['kullanilan'])} belge -> {TARGET_DIR}")
    print(f"Dışarıda bırakılan: {len(catalog['disarida_birakilan'])} dosya")
    print(f"Katalog: {CATALOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
