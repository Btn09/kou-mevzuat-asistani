#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Çok belgeli aramayı (rag_v2) elle hazırlanmış soru -> beklenen madde listesiyle ölçer.

Reranker'lı ve reranker'sız sonuçları karşılaştırır. Model (LLM) kullanılmaz.

Kullanım:
    python -X utf8 arama_testi_v2.py
"""

from __future__ import annotations

import sys

from rag_v2 import Retriever

# (soru, beklenen belge, beklenen madde etiketi). Öğrenci diliyle; farklı belgelere dağılmış.
CASES = [
    ("kaydımı dondurabilir miyim", "kou_onlisans_lisans_egitim_ogretim_yonetmeligi_2019", "MADDE 39"),
    ("izin almak için ne zaman başvurmam lazım", "kou_onlisans_lisans_egitim_ogretim_yonetmeligi_2019", "MADDE 39"),
    ("yatay geçiş başvuru tarihleri ne zaman", "kou_basariya_dayali_yatay_gecis_kilavuzu_2026_2027", "KISIM 1"),
    ("kurum içi yatay geçiş için ortalamam kaç olmalı", "kou_basariya_dayali_yatay_gecis_kilavuzu_2026_2027", "KISIM 2"),
    ("bilgisayar mühendisliğine yatay geçiş kontenjanı kaç", "kou_basariya_dayali_yatay_gecis_kilavuzu_2026_2027", "KISIM 11"),
    ("kopya çekersem ne ceza alırım", "kanun_2547_yuksekogretim", "MADDE 54"),
    ("disiplin cezasına nasıl itiraz ederim", "kanun_2547_yuksekogretim", "MADDE 54"),
    # Yandal: M18 genel tanım, M19 başvuru şartı; ikisi de doğru cevap.
    ("yandal yapabilir miyim", "yok_gecis_cift_anadal_yandal_kredi_transferi_yonetmeligi", ("MADDE 18", "MADDE 19")),
    ("yaz okulunda kaç ders alabilirim", "kou_yaz_ogretimi_esaslari", "ESAS 2"),
    ("başka üniversitenin yaz okulundan ders almak için puan şartı var mı", "kou_yaz_ogretimi_esaslari", "ESAS 1"),
    ("çan eğrisi nasıl uygulanıyor", "kou_basari_notu_degerlendirme_usul_esaslar", "MADDE 10"),
    ("finalde en az kaç almam lazım", "kou_basari_notu_degerlendirme_usul_esaslar", "MADDE 8"),
    ("zorunlu staj nasıl yapılır", "yok_uygulamali_egitimler_cerceve_yonetmeligi", "MADDE 13"),
    ("muafiyet başvurusu ne zaman yapılır", "kou_muafiyet_ve_intibak_yonergesi", "MADDE 6"),
    ("hazırlıktan muaf olmak için TOEFL kaç almalıyım", "kou_yabanci_diller_yuksekokulu_egitim_sinav_yonergesi", "MADDE 11"),
    ("ortalamam 2 üstündeyse en fazla kaç AKTS alabilirim", "kou_ogrenci_danismanlik_hizmetleri_yonergesi", "MADDE 8"),
    ("büte kimler girebilir", "kou_onlisans_lisans_egitim_ogretim_yonetmeligi_2019", "MADDE 20"),
    ("devamsızlık sınırı ne kadar", "kou_onlisans_lisans_egitim_ogretim_yonetmeligi_2019", "MADDE 17"),
    ("yüzde on listesine kimler giremez", "kou_yuzde_on_basari_degerlendirme_yonergesi", "MADDE 6"),
    ("tıp fakültesinde devamsızlık sınırı", "kou_tip_fakultesi_egitim_ogretim_sinav_yonetmeligi", "MADDE 14"),
    ("konservatuvarda harf notları nasıl", "kou_devlet_konservatuvari_lisans_egitim_sinav_yonetmeligi", "MADDE 53"),
    ("doktora yeterlik sınavına kaç kez girebilirim", "kou_lisansustu_egitim_ogretim_yonetmeligi_2023", "MADDE 33"),
    ("tezsiz yüksek lisans proje jürisi kaç kişi", "kou_lisansustu_egitim_ogretim_yonetmeligi_2023", "MADDE 29"),
    ("diplomamı başkası alabilir mi", "kou_diploma_diploma_eki_belgeler_yonergesi", "MADDE 9"),
    # ---- Akademik personel ----
    ("ek ders ücreti nasıl hesaplanır", "kou_ders_gorevlendirme_ek_ders_yonergesi", "MADDE 11"),
    ("öğretim üyesinin haftalık zorunlu ders yükü kaç saat", "kou_ders_gorevlendirme_ek_ders_yonergesi", "MADDE 10"),
    ("BAP projesine nasıl başvururum", "kou_bap_uygulama_yonergesi", "MADDE 10"),
    ("BAP projemin süresini uzatabilir miyim", "kou_bap_uygulama_yonergesi", "MADDE 17"),
    ("yurt dışı kongre için görevlendirme nasıl alınır", "kanun_2547_yuksekogretim", "MADDE 39"),
    ("yurt dışı gündelik nasıl belirlenir harcırah", "kanun_6245_harcirah", "MADDE 34"),
    ("doçentliğe atama şartları neler", "kanun_2547_yuksekogretim", ("MADDE 24", "MADDE 25")),
]


def evaluate(retriever: Retriever, label: str) -> None:
    top1 = top4 = 0
    misses = []
    for question, document, article_label in CASES:
        expected = {(document, label) for label in ([article_label] if isinstance(article_label, str) else article_label)}
        got = [(a["ozellikler"]["belge_id"], a["ozellikler"]["madde_etiketi"]) for a in retriever.retrieve(question)]
        top1 += bool(got) and got[0] in expected
        top4 += bool(expected & set(got))
        if not expected & set(got):
            misses.append(f"{question}  ->  {[g[1] for g in got]}")
    print(f"\n=== {label}: ilk sırada {top1}/{len(CASES)}, ilk 4 içinde {top4}/{len(CASES)}")
    for miss in misses:
        print("   KAÇTI:", miss)


def main() -> int:
    evaluate(Retriever(use_reranker=False), "sadece embedding")
    evaluate(Retriever(use_reranker=True), "embedding + reranker")
    return 0


if __name__ == "__main__":
    sys.exit(main())
