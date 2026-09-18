#!/usr/bin/env python3
"""
eksperimen_atenuasi.py
======================
Menguji SATU hipotesis pada satu stasiun (acemc):

    Model yang dilatih dengan MSE secara sistematis MERATAKAN kejadian
    ekstrem, sehingga RMSE keseluruhan yang baik tidak berarti kinerja
    baik pada episode kritis.

Ini bukan upaya mengejar kinerja terbaik. Kalau hipotesis tidak terbukti,
itu hasil yang sama berharganya dan dilaporkan apa adanya.

    python eksperimen_atenuasi.py hasil --tahap 3
    python eksperimen_atenuasi.py hasil --tahap 3 --seed 7

TAHAP (setiap tahap menjalankan tahap sebelumnya lebih dulu)
  1  muat data, buat jendela, statistik pembagian, cakupan musim, status QAQC
  2  persistence dan seasonal naive
  3  DLinear dilatih dengan MSE, dibandingkan dengan kedua baseline
  4  gambar dan tabel akhir                             (belum dibuat)

RANCANGAN
  Target univariat DO_mgl. Input 288 langkah (72 jam), horizon 96 langkah
  (24 jam), prediksi langsung 96 nilai sekaligus.

  Pembagian kronologis 70/10/20 dilakukan pada GRID WAKTU, bukan pada daftar
  jendela. Setiap jendela (input + target) harus utuh di dalam satu bagian,
  jadi tidak ada jendela yang melintasi batas latih/validasi/uji. Harganya
  383 titik awal jendela hilang di setiap batas, jumlah yang kecil dibanding
  risiko kebocoran.

  Jendela digeser satu langkah (stride 1). Jendela yang memuat NaN di input
  maupun target dibuang, tidak diinterpolasi, dan jumlahnya dilaporkan.

JENDELA EKSTREM
  Jendela yang 96 langkah horizon aktualnya beririsan dengan episode D1 dari
  hasil/288653/episode/acemc_D1.csv. Episode dipetakan ke grid dengan
  tandai_episode dari cakupan_waktu.py, sehingga definisinya sama persis
  dengan tabel perbandingan_cakupan.csv.

AGREGASI DUA TINGKAT
  Dengan stride 1, jendela bertetangga hampir identik dan satu episode
  menyumbang hingga ratusan jendela. Median atas jendela akan didominasi
  episode terpanjang. Maka rasio atenuasi dan galat amplitudo puncak pada
  jendela ekstrem diringkas dua tingkat:
      nilai per jendela -> median di dalam episode -> median lintas episode
  dengan n = jumlah episode. Median tingkat jendela tetap dilaporkan sebagai
  pembanding, dengan akhiran _tingkat_jendela.

  Horizon 24 jam bisa menyentuh lebih dari satu episode, karena hipoksia
  diurnal berulang setiap hari. Jendela seperti itu ditetapkan ke episode
  yang memuat DO AKTUAL TERENDAH di dalam horizonnya, sejalan dengan galat
  amplitudo puncak yang juga berpatokan pada minimum aktual. Banyaknya
  jendela yang menyentuh lebih dari satu episode dilaporkan.

  Rasio atenuasi dan galat puncak jendela NORMAL tetap tingkat jendela sesuai
  spesifikasi, karena jendela normal tidak punya episode pengelompok.

PENYIMPANGAN YANG DITERIMA
  Bagian uji tidak memuat musim panas yang penuh. Musim panas 2026 tercakup
  95,2% karena data acemc berakhir 27 Agustus 2026, dan diterima sebagai
  memenuhi syarat (keputusan (a)). Musim panas 2025 hanya terisi 43,2%, jadi
  menggeser batas uji ke Juni 2025 menukar musim panas yang baik dengan yang
  buruk sambil merusak validasi.

  Bagian validasi (Des 2024 - Jul 2025) miskin kejadian ekstrem, sehingga
  pemilihan epoch DLinear ditentukan terutama oleh perilaku di luar musim
  panas. Ini DISENGAJA: begitulah praktik standar yang sedang diuji.

BATASAN: STATUS QAQC TIDAK SERAGAM
  Data 2021-2024 berstatus final (Historical=1). Data 2025 hingga 24 Juni
  2026 sudah melewati QAQC sekunder (ProvisionalPlus=1). Setelah itu data
  masih mentah-sementara (ProvisionalPlus=0): tidak ada satu pun nilai yang
  ditolak atau ditandai hilang. Sebagian besar episode D1 di bagian uji jatuh
  di periode itu. Tahap 1 mencetak rinciannya. Ini dicatat sebagai batasan,
  eksperimen tidak dihentikan.

ANALISIS SEKUNDER: RINCIAN MENURUT STATUS QAQC
  Ditetapkan sebelum hasil DLinear terlihat. Untuk setiap model: RMSE
  keseluruhan, RMSE ekstrem, dan rasio atenuasi tingkat episode, terpisah
  untuk final (2021-2024), ProvisionalPlus=1, dan ProvisionalPlus=0. Jendela
  ekstrem ikut status episode pemiliknya, jendela normal ikut langkah pertama
  horizonnya. Kelompok < 10 episode dilaporkan apa adanya dengan peringatan.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

from cakupan_waktu import tandai_episode
from siapkan_data_nerrs import kode_flag

DATASET = "288653"
STASIUN = "acemc"
KOLOM_DO = "DO_mgl"
KOLOM_WAKTU = "DateTimeStamp"
FREKUENSI = "15min"

PANJANG_INPUT = 288        # 72 jam
HORIZON = 96               # 24 jam
PORSI = {"latih": 0.7, "validasi": 0.1, "uji": 0.2}
SEED = 42

BULAN_MUSIM_PANAS = (6, 7, 8)
# musim panas di bagian uji yang tidak penuh tetapi DITERIMA atas keputusan
# eksplisit, bukan dilonggarkan diam-diam; lihat docstring
MUSIM_PANAS_DITERIMA = {
    2026: "tercakup 95,2%, data acemc berakhir 27 Ags 2026; keputusan (a)",
}
MIN_JENDELA_EKSTREM = 30

STATUS_QAQC = {"final": "Historical=1", "provisional_plus": "ProvisionalPlus=1",
               "provisional": "ProvisionalPlus=0"}


# ----------------------------------------------------------------------------
# 1. DATA DAN JENDELA
# ----------------------------------------------------------------------------
def muat_deret(akar):
    """DO acemc pada grid 15 menit yang teratur, NaN dibiarkan apa adanya."""
    p = os.path.join(akar, DATASET, "bersih", f"{STASIUN}.csv")
    if not os.path.exists(p):
        sys.exit(f"Berkas bersih '{p}' tidak ada. Jalankan jalankan_semua.py dulu.")
    df = pd.read_csv(p, parse_dates=[KOLOM_WAKTU], index_col=KOLOM_WAKTU,
                     usecols=[KOLOM_WAKTU, KOLOM_DO])
    # berkas bersih sudah di-resample, asfreq hanya menjamin tidak ada langkah
    # yang terlewat sebelum indeks posisi dipakai sebagai jarak waktu
    return df[KOLOM_DO].asfreq(FREKUENSI)


def muat_episode(akar):
    p = os.path.join(akar, DATASET, "episode", f"{STASIUN}_D1.csv")
    if not os.path.exists(p):
        sys.exit(f"Berkas episode '{p}' tidak ada. Jalankan jalankan_semua.py dulu.")
    return pd.read_csv(p, parse_dates=["mulai", "selesai"])


def batas_bagian(n):
    """Posisi [awal, akhir) tiap bagian pada grid, urut kronologis."""
    batas, awal = {}, 0
    for i, (nama, porsi) in enumerate(PORSI.items()):
        akhir = n if i == len(PORSI) - 1 else awal + int(round(porsi * n))
        batas[nama] = (awal, akhir)
        awal = akhir
    return batas


def jumlah_dalam_rentang(kumulatif, awal, panjang):
    """Banyaknya True pada [awal, awal + panjang) untuk banyak awal sekaligus."""
    return kumulatif[awal + panjang] - kumulatif[awal]


def buat_jendela(nilai, awal, akhir):
    """Posisi awal jendela yang utuh di [awal, akhir) dan bebas NaN.

    Kembaliannya (posisi_valid, statistik). Tidak ada nilai yang diisi.
    """
    total = PANJANG_INPUT + HORIZON
    kandidat = np.arange(awal, akhir - total + 1)
    nan_kum = np.concatenate(([0], np.cumsum(np.isnan(nilai))))
    nan_input = jumlah_dalam_rentang(nan_kum, kandidat, PANJANG_INPUT) > 0
    nan_target = jumlah_dalam_rentang(nan_kum, kandidat + PANJANG_INPUT,
                                      HORIZON) > 0
    buang = nan_input | nan_target
    stat = {
        "kandidat": len(kandidat),
        "valid": int((~buang).sum()),
        "buang_NaN": int(buang.sum()),
        "buang_input_saja": int((nan_input & ~nan_target).sum()),
        "buang_target_saja": int((~nan_input & nan_target).sum()),
        "buang_keduanya": int((nan_input & nan_target).sum()),
    }
    return kandidat[~buang], stat


def ambil(nilai, posisi, geser, panjang):
    """Matriks (jendela x panjang) dari nilai[posisi + geser : + panjang]."""
    return nilai[posisi[:, None] + geser + np.arange(panjang)]


def nomor_episode_grid(indeks, episode):
    """Nomor episode per langkah grid, -1 di luar episode."""
    nomor = np.full(len(indeks), -1)
    a = indeks.searchsorted(episode["mulai"].to_numpy(), side="left")
    b = indeks.searchsorted(episode["selesai"].to_numpy(), side="right")
    for k, (i, j) in enumerate(zip(a, b)):
        nomor[i:j] = k
    return nomor


def episode_per_jendela(indeks, nilai, episode, posisi):
    """Nomor episode tiap jendela: -1 = normal, >= 0 = ekstrem.

    Horizon yang menyentuh lebih dari satu episode ditetapkan ke episode yang
    memuat DO aktual terendah di dalam horizon. Satu jendela tidak pernah
    dihitung di beberapa episode. Kembalian kedua dan ketiga: banyaknya
    jendela yang menyentuh lebih dari satu episode, dan banyaknya episode
    berbeda yang tersentuh horizon mana pun (termasuk yang tidak memiliki
    jendela sendiri karena berdekatan dengan episode yang lebih dalam).
    """
    dalam = tandai_episode(indeks, episode)
    kum = np.concatenate(([0], np.cumsum(dalam)))
    ekstrem = jumlah_dalam_rentang(kum, posisi + PANJANG_INPUT, HORIZON) > 0

    hasil = np.full(len(posisi), -1)
    if not ekstrem.any():
        return hasil, 0, 0
    nomor = ambil(nomor_episode_grid(indeks, episode), posisi[ekstrem],
                  PANJANG_INPUT, HORIZON)
    aktual = ambil(nilai, posisi[ekstrem], PANJANG_INPUT, HORIZON)
    terendah = np.where(nomor >= 0, aktual, np.inf).argmin(axis=1)
    hasil[ekstrem] = nomor[np.arange(len(nomor)), terendah]

    nomor_min = np.where(nomor >= 0, nomor, np.iinfo(nomor.dtype).max).min(axis=1)
    ganda = int((nomor.max(axis=1) != nomor_min).sum())
    tersentuh = len(np.unique(nomor[nomor >= 0]))
    return hasil, ganda, tersentuh


def cakupan_musim_panas(seri, awal, akhir):
    """Untuk tiap tahun yang musim panasnya menyentuh bagian ini: apakah
    1 Jun 00:00 s/d 31 Ags 23:45 seluruhnya berada di dalam bagian, dan
    berapa persen titik DO-nya terisi."""
    potong = seri.iloc[awal:akhir]
    t0, t1 = potong.index[0], potong.index[-1]
    baris = []
    for tahun in range(t0.year, t1.year + 1):
        mulai = pd.Timestamp(tahun, BULAN_MUSIM_PANAS[0], 1)
        selesai = pd.Timestamp(tahun, BULAN_MUSIM_PANAS[-1] + 1, 1) - pd.Timedelta(FREKUENSI)
        if selesai < t0 or mulai > t1:
            continue
        bagian = potong[max(mulai, t0):min(selesai, t1)]
        n_penuh = len(pd.date_range(mulai, selesai, freq=FREKUENSI))
        baris.append({
            "tahun": tahun,
            "tercakup_dari": bagian.index[0],
            "tercakup_sampai": bagian.index[-1],
            "penuh": bool(mulai >= t0 and selesai <= t1),
            "persen_kalender": round(100 * len(bagian) / n_penuh, 1),
            "persen_DO_terisi": round(100 * bagian.notna().sum() / n_penuh, 1),
            "diterima": MUSIM_PANAS_DITERIMA.get(tahun, ""),
        })
    return pd.DataFrame(baris)


# ----------------------------------------------------------------------------
# 2. STATUS QAQC PER TAHUN
# ----------------------------------------------------------------------------
def muat_status_qaqc(dir_mentah):
    """Status QAQC dan kode flag DO per baris dari berkas mentah tahunan.

    Berkas bersih sudah membuang kolom status, jadi yang dibaca berkas mentah.
    Kembaliannya None kalau data mentah tidak tersedia (tidak ikut repo).
    """
    pola = re.compile(rf"^{STASIUN}wq\d{{4}}\.csv$", re.I)
    berkas = [b for b in sorted(glob.glob(os.path.join(dir_mentah, "*.csv")))
              if pola.match(os.path.basename(b))]
    if not berkas:
        return None
    potongan = []
    for b in berkas:
        d = pd.read_csv(b, usecols=[KOLOM_WAKTU, "Historical", "ProvisionalPlus",
                                    "F_" + KOLOM_DO], low_memory=False)
        d[KOLOM_WAKTU] = pd.to_datetime(d[KOLOM_WAKTU], format="%m/%d/%Y %H:%M",
                                        errors="coerce")
        potongan.append(d.dropna(subset=[KOLOM_WAKTU]))
    d = pd.concat(potongan, ignore_index=True).set_index(KOLOM_WAKTU).sort_index()
    d = d[~d.index.duplicated(keep="first")]
    d["status"] = np.select(
        [d["Historical"] == 1, d["ProvisionalPlus"] == 1],
        ["final", "provisional_plus"], default="provisional")
    d["flag"] = kode_flag(d["F_" + KOLOM_DO])
    return d[["status", "flag"]]


def laporan_qaqc(status, seri, episode, posisi_uji, ep_uji):
    print("\nSTATUS QAQC PER TAHUN (berkas mentah)")
    if status is None:
        print("  data mentah tidak tersedia, pemeriksaan dilewati.")
        return
    tahun = status.index.year
    t_status = pd.crosstab(tahun, status["status"], normalize="index").mul(100).round(1)
    t_status = t_status.reindex(columns=list(STATUS_QAQC), fill_value=0.0)
    t_flag = pd.crosstab(tahun, status["flag"], normalize="index").mul(100).round(2)
    t_flag.columns = [f"flag_{int(c)}_%" for c in t_flag.columns]
    tabel = pd.concat([t_status.add_suffix("_%"), t_flag], axis=1)
    tabel.index.name = "tahun"
    print(tabel.to_string())
    print("  " + ", ".join(f"{k} = {v}" for k, v in STATUS_QAQC.items()))

    prov = status.index[status["status"] == "provisional"]
    if len(prov):
        print(f"\n  Periode provisional (belum QAQC sekunder): {prov.min()}  s/d  "
              f"{prov.max()}")
        tolak = status.loc[prov, "flag"].ne(0).sum()
        print(f"  Nilai DO bukan <0> di periode itu: {tolak:,} dari {len(prov):,}")

    # status tiap episode diambil dari langkah mulainya
    st_grid = status["status"].reindex(seri.index)
    episode = episode.assign(status=st_grid.reindex(episode["mulai"]).to_numpy())
    dipakai = np.unique(ep_uji[ep_uji >= 0])
    baris = episode.iloc[dipakai]
    per_jendela = pd.Series(episode["status"].to_numpy()[ep_uji[ep_uji >= 0]])
    ringkas = pd.DataFrame({
        "episode_D1": baris["status"].value_counts(),
        "jam_episode": baris.groupby("status")["durasi_jam"].sum(),
        "jendela_ekstrem": per_jendela.value_counts(),
    }).reindex(list(STATUS_QAQC)).fillna(0).astype({"episode_D1": int,
                                                    "jendela_ekstrem": int})
    ringkas.index.name = "status"
    print("\n  Episode D1 dan jendela ekstrem di bagian uji menurut status QAQC:")
    print("  " + ringkas.to_string().replace("\n", "\n  "))
    porsi = ringkas.loc["provisional", "episode_D1"] / max(1, ringkas["episode_D1"].sum())
    if porsi > 0:
        print(f"\n  BATASAN: {100 * porsi:.0f}% episode D1 uji berasal dari data yang "
              f"belum melewati QAQC sekunder.")
        print("  Episode itu bisa memuat pembacaan rendah akibat alat yang pada data")
        print("  final akan ditolak. Eksperimen tetap dilanjutkan.")


def tahap_1(akar, dir_mentah):
    seri = muat_deret(akar)
    episode = muat_episode(akar)
    nilai = seri.to_numpy(dtype=float)
    batas = batas_bagian(len(seri))

    print("\n" + "=" * 92)
    print(f"TAHAP 1 — DATA DAN JENDELA  ({DATASET}/{STASIUN})")
    print("=" * 92)
    print(f"  Rentang grid        : {seri.index[0]}  s/d  {seri.index[-1]}")
    print(f"  Titik grid          : {len(seri):,}  (DO terisi {seri.notna().sum():,}, "
          f"{100 * seri.notna().mean():.1f}%)")
    print(f"  Episode D1          : {len(episode)}")
    print(f"  Jendela             : input {PANJANG_INPUT} + horizon {HORIZON} langkah, "
          f"stride 1")

    baris, jendela = [], {}
    for nama, (awal, akhir) in batas.items():
        posisi, stat = buat_jendela(nilai, awal, akhir)
        ep, ganda, tersentuh = episode_per_jendela(seri.index, nilai, episode, posisi)
        jendela[nama] = (posisi, ep)
        n_ekstrem = int((ep >= 0).sum())
        n_pemilik = len(np.unique(ep[ep >= 0]))
        baris.append({
            "bagian": nama,
            "mulai": seri.index[awal],
            "selesai": seri.index[akhir - 1],
            "titik_grid": akhir - awal,
            **stat,
            "persen_buang": round(100 * stat["buang_NaN"] / stat["kandidat"], 1),
            "jendela_ekstrem": n_ekstrem,
            "persen_ekstrem": round(100 * n_ekstrem / len(ep), 1) if len(ep) else None,
            "episode_D1": n_pemilik,
            "episode_tersentuh": tersentuh,
            "episode_tak_terwakili": tersentuh - n_pemilik,
            "jendela_multi_episode": ganda,
        })
    tabel = pd.DataFrame(baris)

    print("\nPEMBAGIAN KRONOLOGIS 70/10/20")
    print(tabel[["bagian", "mulai", "selesai", "titik_grid"]].to_string(index=False))
    print("\nJENDELA PER BAGIAN")
    print(tabel[["bagian", "kandidat", "valid", "buang_NaN", "persen_buang",
                 "buang_input_saja", "buang_target_saja", "buang_keduanya"]]
          .to_string(index=False))
    print("\nJENDELA EKSTREM (horizon beririsan dengan episode D1)")
    print(tabel[["bagian", "valid", "jendela_ekstrem", "persen_ekstrem",
                 "episode_D1", "episode_tersentuh", "episode_tak_terwakili",
                 "jendela_multi_episode"]].to_string(index=False))
    print("  episode_D1 = episode yang memiliki minimal satu jendela (n pada metrik)")

    uji = tabel.set_index("bagian").loc["uji"]
    print(f"\n  {uji['episode_tak_terwakili']} dari {uji['episode_tersentuh']} episode D1 "
          f"yang tersentuh horizon uji TIDAK TERWAKILI:")
    print("  setiap horizon yang menyentuhnya juga menyentuh episode lain yang lebih")
    print("  dalam, dan jendela ditetapkan ke episode terdalam itu. Satu jendela tidak")
    print("  dihitung di beberapa episode, supaya n episode tetap saling lepas.")
    if uji["jendela_ekstrem"] < MIN_JENDELA_EKSTREM:
        print(f"\n  PERINGATAN: jendela ekstrem di bagian uji < {MIN_JENDELA_EKSTREM}. "
              f"Metrik ekstrem tidak dapat diandalkan.")
    if uji["episode_D1"] < MIN_JENDELA_EKSTREM:
        print(f"\n  PERINGATAN: jendela ekstrem uji berasal dari hanya "
              f"{uji['episode_D1']} episode D1. Metrik tingkat episode tidak "
              f"dapat diandalkan.")

    print("\nCAKUPAN MUSIM PANAS (Jun-Ags) DI BAGIAN UJI")
    awal, akhir = batas["uji"]
    musim = cakupan_musim_panas(seri, awal, akhir)
    print(musim.to_string(index=False))
    if len(musim) and musim["penuh"].any():
        lolos = True
        print("\n  Bagian uji memuat minimal satu musim panas penuh. Syarat terpenuhi.")
    elif len(musim) and (musim["diterima"] != "").any():
        lolos = True
        print("\n  PENYIMPANGAN: tidak ada musim panas penuh di bagian uji. Diterima atas")
        print("  keputusan eksplisit, lihat kolom 'diterima' dan README.")
    else:
        lolos = False
        print("\n  BERHENTI: bagian uji TIDAK memuat satu pun musim panas penuh.")
        print("  Tahap berikutnya tidak dijalankan sebelum ada keputusan.")

    status = muat_status_qaqc(dir_mentah)
    laporan_qaqc(status, seri, episode, *jendela["uji"])
    return {"seri": seri, "nilai": nilai, "batas": batas, "jendela": jendela,
            "episode": episode, "lolos": lolos, "tabel_jendela": tabel,
            "status_grid": None if status is None
            else status["status"].reindex(seri.index).to_numpy()}


# ----------------------------------------------------------------------------
# 3. METRIK
# ----------------------------------------------------------------------------
def median_aman(x):
    """Median yang mengabaikan NaN dan mengembalikan NaN untuk isian kosong."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(np.median(x)) if len(x) else np.nan


def median_dua_tingkat(nilai_jendela, nomor_episode):
    """Median di dalam episode, lalu median lintas episode. NaN diabaikan."""
    s = pd.Series(nilai_jendela).groupby(nomor_episode).median().dropna()
    return (float(s.median()) if len(s) else np.nan), len(s)


def hitung_metrik(nama, aktual, prediksi, ep):
    """Semua metrik pada skala asli (mg/L). ep: nomor episode, -1 = normal."""
    galat = prediksi - aktual
    kuadrat = galat ** 2
    ekstrem = ep >= 0

    def rmse_mae(m):
        if not m.any():
            return np.nan, np.nan
        return float(np.sqrt(kuadrat[m].mean())), float(np.abs(galat[m]).mean())

    rentang_aktual = aktual.max(axis=1) - aktual.min(axis=1)
    rentang_pred = prediksi.max(axis=1) - prediksi.min(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rasio = np.where(rentang_aktual > 0, rentang_pred / rentang_aktual, np.nan)
    galat_puncak = prediksi.min(axis=1) - aktual.min(axis=1)

    rasio_ep, n_ep = median_dua_tingkat(rasio[ekstrem], ep[ekstrem])
    puncak_ep, _ = median_dua_tingkat(galat_puncak[ekstrem], ep[ekstrem])
    sse = kuadrat.sum(axis=1)

    hasil = {"model": nama, "n_jendela_uji": len(ep),
             "n_jendela_ekstrem": int(ekstrem.sum()),
             "n_jendela_normal": int((~ekstrem).sum()),
             "n_episode_ekstrem": n_ep}
    for label, m in (("semua", np.ones_like(ekstrem)), ("ekstrem", ekstrem),
                     ("normal", ~ekstrem)):
        hasil[f"rmse_{label}"], hasil[f"mae_{label}"] = rmse_mae(m)
    hasil.update({
        "rasio_atenuasi_ekstrem": rasio_ep,
        "rasio_atenuasi_ekstrem_tingkat_jendela": median_aman(rasio[ekstrem]),
        "rasio_atenuasi_normal_tingkat_jendela": median_aman(rasio[~ekstrem]),
        "galat_puncak_ekstrem": puncak_ep,
        "galat_puncak_ekstrem_tingkat_jendela": median_aman(galat_puncak[ekstrem]),
        "persen_SSE_dari_ekstrem": 100 * sse[ekstrem].sum() / sse.sum(),
        "persen_jendela_ekstrem": 100 * ekstrem.mean(),
        "n_rasio_tak_terdefinisi": int(np.isnan(rasio).sum()),
    })
    return hasil


def cetak_metrik(tabel):
    f = lambda x: f"{x:.3f}"
    t = tabel.set_index("model")
    print("\n  n jendela uji: {:,} (ekstrem {:,}, normal {:,}); n episode ekstrem: {}".format(
        *t.iloc[0][["n_jendela_uji", "n_jendela_ekstrem", "n_jendela_normal",
                    "n_episode_ekstrem"]].astype(int)))
    print("\nGALAT (mg/L)")
    print(t[["rmse_semua", "mae_semua", "rmse_ekstrem", "mae_ekstrem",
             "rmse_normal", "mae_normal"]].to_string(float_format=f))
    print("\nRASIO ATENUASI (1 = amplitudo sesuai, < 1 = meratakan)")
    print(t[["rasio_atenuasi_ekstrem", "rasio_atenuasi_ekstrem_tingkat_jendela",
             "rasio_atenuasi_normal_tingkat_jendela"]].to_string(float_format=f))
    print("  rasio_atenuasi_ekstrem = median lintas episode dari median per episode")
    print("\nGALAT AMPLITUDO PUNCAK (mg/L, positif = tidak turun sedalam aktual)")
    print(t[["galat_puncak_ekstrem", "galat_puncak_ekstrem_tingkat_jendela"]]
          .to_string(float_format=f))
    print("\nKONTRIBUSI KUADRAT GALAT")
    print(t[["persen_SSE_dari_ekstrem", "persen_jendela_ekstrem"]]
          .to_string(float_format=lambda x: f"{x:.1f}"))
    tak = t["n_rasio_tak_terdefinisi"]
    if tak.any():
        print(f"\n  {int(tak.max())} jendela berentang aktual nol, rasionya tak terdefinisi "
              f"dan diabaikan.")


# ----------------------------------------------------------------------------
# 4. ANALISIS SEKUNDER: RINCIAN MENURUT STATUS QAQC
# ----------------------------------------------------------------------------
# Ditetapkan SEBELUM hasil DLinear terlihat. Aturan pengelompokan:
#   jendela ekstrem  -> status episode pemiliknya, diambil di langkah mulai
#   jendela normal   -> status langkah pertama horizon
# Kelompok dengan < MIN_EPISODE_KELOMPOK episode dilaporkan apa adanya dengan
# peringatan, tidak digabung ke kelompok lain.
MIN_EPISODE_KELOMPOK = 10
LABEL_STATUS = {"final": "final (2021-2024)",
                "provisional_plus": "ProvisionalPlus=1",
                "provisional": "ProvisionalPlus=0"}


def status_per_jendela(ctx, posisi, ep):
    """Status QAQC tiap jendela uji. Kembalian kedua: banyaknya jendela yang
    horizonnya melintasi batas status."""
    st = ctx["status_grid"]
    st_episode = st[ctx["seri"].index.searchsorted(ctx["episode"]["mulai"].to_numpy())]
    awal_h = posisi + PANJANG_INPUT
    hasil = st[awal_h].astype(object)
    hasil[ep >= 0] = st_episode[ep[ep >= 0]]
    lintas = int((st[awal_h] != st[awal_h + HORIZON - 1]).sum())
    return hasil, lintas


def rincian_qaqc(ctx, aktual, prediksi, ep):
    if ctx["status_grid"] is None:
        print("\n  Data mentah tidak tersedia, rincian status QAQC dilewati.")
        return None
    status, lintas = status_per_jendela(ctx, ctx["jendela"]["uji"][0], ep)
    baris = []
    for model, p in prediksi.items():
        for s, label in LABEL_STATUS.items():
            m = status == s
            r = {"model": model, "status": label, "n_jendela": int(m.sum()),
                 "n_jendela_ekstrem": 0, "n_episode_ekstrem": 0,
                 "rmse_semua": np.nan, "rmse_ekstrem": np.nan,
                 "rasio_atenuasi_ekstrem": np.nan}
            if m.any():
                h = hitung_metrik(model, aktual[m], p[m], ep[m])
                r.update({k: h[k] for k in ("n_jendela_ekstrem", "n_episode_ekstrem",
                                            "rmse_semua", "rmse_ekstrem",
                                            "rasio_atenuasi_ekstrem")})
            baris.append(r)
    tabel = pd.DataFrame(baris)

    print("\n" + "-" * 92)
    print("ANALISIS SEKUNDER (ditetapkan sebelum hasil DLinear): RINCIAN STATUS QAQC")
    print("-" * 92)
    print(tabel.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("  Jendela ekstrem dikelompokkan menurut status episode pemiliknya, jendela")
    print(f"  normal menurut langkah pertama horizon. {lintas} jendela horizonnya "
          f"melintasi batas status.")
    kelompok = tabel.drop_duplicates("status")
    for _, r in kelompok[kelompok["n_episode_ekstrem"] < MIN_EPISODE_KELOMPOK].iterrows():
        if r["n_jendela"] == 0:
            print(f"  PERINGATAN: kelompok {r['status']} KOSONG di bagian uji (0 jendela, "
                  f"0 episode), tidak ada metrik.")
        else:
            print(f"  PERINGATAN: kelompok {r['status']} hanya {r['n_episode_ekstrem']} "
                  f"episode (< {MIN_EPISODE_KELOMPOK}), metriknya tidak dapat diandalkan.")
    return tabel


def evaluasi(ctx, prediksi):
    """Metrik utama dan rincian QAQC untuk semua model yang sudah ada."""
    posisi, ep = ctx["jendela"]["uji"]
    aktual = ambil(ctx["nilai"], posisi, PANJANG_INPUT, HORIZON)
    tabel = pd.DataFrame([hitung_metrik(n, aktual, p, ep) for n, p in prediksi.items()])
    cetak_metrik(tabel)
    uji = ctx["tabel_jendela"].set_index("bagian").loc["uji"]
    print(f"\n  n episode = {uji['episode_D1']}. {uji['episode_tak_terwakili']} episode "
          f"lain tersentuh horizon uji tetapi tidak terwakili,")
    print("  karena berdekatan dengan episode yang lebih dalam.")
    ctx["metrik"] = tabel
    ctx["rincian_qaqc"] = rincian_qaqc(ctx, aktual, prediksi, ep)
    return ctx


def tahap_2(ctx):
    posisi, _ = ctx["jendela"]["uji"]
    masukan = ambil(ctx["nilai"], posisi, 0, PANJANG_INPUT)
    ctx["prediksi"] = {
        "persistence": np.repeat(masukan[:, -1:], HORIZON, axis=1),
        "seasonal_naive": masukan[:, -HORIZON:],
    }

    print("\n" + "=" * 92)
    print("TAHAP 2 — BASELINE: PERSISTENCE DAN SEASONAL NAIVE (bagian uji)")
    print("=" * 92)
    evaluasi(ctx, ctx["prediksi"])
    print("\n  Rasio atenuasi persistence bernilai 0 menurut konstruksinya (prediksi")
    print("  datar), bukan temuan. Seasonal naive menyalin amplitudo hari sebelumnya.")
    return ctx


# ----------------------------------------------------------------------------
# 5. DLINEAR
# ----------------------------------------------------------------------------
# Hiperparameter mengikuti bawaan repositori DLinear (Zeng dkk., 2023) dan
# TIDAK disetel pada data ini: yang diuji adalah praktik standar, bukan model
# terbaik. Loss MSE tanpa pembobotan, disengaja.
KERNEL_MA = 25
UKURAN_BATCH = 32
LAJU_BELAJAR = 0.005       # dibagi dua setiap epoch, seperti lradj type1
MAKS_EPOCH = 10
KESABARAN = 3              # early stopping pada MSE validasi
UKURAN_POTONG_EVAL = 4096


def buat_dlinear(torch, nn):
    class DLinear(nn.Module):
        """Rata-rata bergerak memisahkan tren dan sisa, masing-masing satu
        lapis linear 288 -> 96, lalu dijumlahkan. Tepi diisi nilai ujung."""

        def __init__(self):
            super().__init__()
            self.rata = nn.AvgPool1d(KERNEL_MA, stride=1)
            self.linear_tren = nn.Linear(PANJANG_INPUT, HORIZON)
            self.linear_sisa = nn.Linear(PANJANG_INPUT, HORIZON)

        def forward(self, x):
            pad = (KERNEL_MA - 1) // 2
            tepi = torch.cat([x[:, :1].repeat(1, pad), x,
                              x[:, -1:].repeat(1, KERNEL_MA - 1 - pad)], dim=1)
            tren = self.rata(tepi.unsqueeze(1)).squeeze(1)
            return self.linear_tren(tren) + self.linear_sisa(x - tren)

    return DLinear()


def latih_dlinear(ctx, seed):
    # torch diimpor di sini supaya tahap 1-2 tetap jalan tanpa torch
    import time

    import torch
    from torch import nn

    torch.manual_seed(seed)
    awal, akhir = ctx["batas"]["latih"]
    # normalisasi HANYA dari bagian latih, lalu dipakai untuk validasi dan uji
    rerata = float(np.nanmean(ctx["nilai"][awal:akhir]))
    sb = float(np.nanstd(ctx["nilai"][awal:akhir]))
    z = torch.tensor((ctx["nilai"] - rerata) / sb, dtype=torch.float32)
    geser_x = torch.arange(PANJANG_INPUT)
    geser_y = torch.arange(PANJANG_INPUT, PANJANG_INPUT + HORIZON)

    def batch(pos):
        p = torch.as_tensor(pos)[:, None]
        return z[p + geser_x], z[p + geser_y]

    def ramal(model, pos):
        model.eval()
        keluar = []
        with torch.no_grad():
            for i in range(0, len(pos), UKURAN_POTONG_EVAL):
                keluar.append(model(batch(pos[i:i + UKURAN_POTONG_EVAL])[0]))
        return torch.cat(keluar)

    pos_latih = ctx["jendela"]["latih"][0]
    pos_val = ctx["jendela"]["validasi"][0]
    y_val = batch(pos_val)[1]

    model = buat_dlinear(torch, nn)
    opt = torch.optim.Adam(model.parameters(), lr=LAJU_BELAJAR)
    mse = nn.MSELoss()
    acak = torch.Generator().manual_seed(seed)

    print(f"\n  torch {torch.__version__}, CPU, {torch.get_num_threads()} thread")
    print(f"  Normalisasi dari bagian latih: rerata {rerata:.4f}, sb {sb:.4f} mg/L")
    print(f"  Jendela latih {len(pos_latih):,}, validasi {len(pos_val):,}; batch "
          f"{UKURAN_BATCH}, lr awal {LAJU_BELAJAR}, maks {MAKS_EPOCH} epoch, "
          f"kesabaran {KESABARAN}, kernel {KERNEL_MA}")
    print(f"\n  {'epoch':>5} {'lr':>9} {'MSE_latih':>10} {'MSE_val':>10} {'detik':>7}")
    riwayat, terbaik, bobot_terbaik, epoch_terbaik, sabar = [], np.inf, None, 0, 0
    mulai_latih = time.perf_counter()
    for epoch in range(1, MAKS_EPOCH + 1):
        lr = LAJU_BELAJAR * 0.5 ** (epoch - 1)
        for g in opt.param_groups:
            g["lr"] = lr
        t0 = time.perf_counter()
        model.train()
        urut = pos_latih[torch.randperm(len(pos_latih), generator=acak).numpy()]
        total, n = 0.0, 0
        for i in range(0, len(urut), UKURAN_BATCH):
            x, y = batch(urut[i:i + UKURAN_BATCH])
            opt.zero_grad()
            loss = mse(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
            n += len(x)
        mse_val = mse(ramal(model, pos_val), y_val).item()
        detik = time.perf_counter() - t0
        riwayat.append({"epoch": epoch, "lr": lr, "mse_latih": total / n,
                        "mse_val": mse_val, "detik": detik})
        print(f"  {epoch:>5} {lr:>9.6f} {total / n:>10.5f} {mse_val:>10.5f} {detik:>7.1f}")
        if mse_val < terbaik:
            terbaik, epoch_terbaik, sabar = mse_val, epoch, 0
            bobot_terbaik = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            sabar += 1
            if sabar >= KESABARAN:
                print(f"  early stopping: MSE validasi tidak membaik {KESABARAN} epoch")
                break
    waktu_latih = time.perf_counter() - mulai_latih
    model.load_state_dict(bobot_terbaik)
    print(f"\n  Waktu latih: {waktu_latih:.1f} detik, {len(riwayat)} epoch, bobot "
          f"terbaik dari epoch {epoch_terbaik} (MSE val ternormalisasi {terbaik:.5f})")

    # kembali ke skala asli sebelum metrik apa pun dihitung
    pred = ramal(model, ctx["jendela"]["uji"][0]).numpy().astype(float) * sb + rerata
    return pred, {"waktu_latih_detik": waktu_latih, "epoch_dijalankan": len(riwayat),
                  "epoch_terbaik": epoch_terbaik, "mse_val_terbaik": terbaik,
                  "riwayat": pd.DataFrame(riwayat)}


def bandingkan(tabel):
    """Pernyataan terus terang DLinear terhadap kedua baseline."""
    t = tabel.set_index("model")
    dl, ps, sn = t.loc["dlinear"], t.loc["persistence"], t.loc["seasonal_naive"]

    print("\n" + "-" * 92)
    print("DLINEAR TERHADAP KEDUA BASELINE")
    print("-" * 92)
    for label, kol in (("RMSE semua", "rmse_semua"), ("RMSE ekstrem", "rmse_ekstrem"),
                       ("RMSE normal", "rmse_normal")):
        for nama, b in (("persistence", ps), ("seasonal naive", sn)):
            print(f"  {label:<13}: DLinear {dl[kol]:.3f} vs {nama:<15} {b[kol]:.3f}  "
                  f"-> DLinear {'MENANG' if dl[kol] < b[kol] else 'KALAH'} "
                  f"({100 * (dl[kol] - b[kol]) / b[kol]:+.1f}%)")
    print(f"  Rasio atenuasi ekstrem (episode): DLinear {dl['rasio_atenuasi_ekstrem']:.3f}"
          f" vs seasonal naive {sn['rasio_atenuasi_ekstrem']:.3f}")
    print(f"  Galat puncak ekstrem (episode)  : DLinear {dl['galat_puncak_ekstrem']:+.2f}"
          f" vs seasonal naive {sn['galat_puncak_ekstrem']:+.2f} mg/L")

    if dl["rmse_semua"] >= ps["rmse_semua"]:
        print("\n  TERUS TERANG: DLinear TIDAK mengalahkan persistence pada RMSE")
        print("  keseluruhan. Model ini tidak berguna sebagai peramal.")
    if dl["rmse_semua"] >= sn["rmse_semua"]:
        print("\n  TERUS TERANG: DLinear TIDAK mengalahkan seasonal naive, lantai yang")
        print("  sesungguhnya, pada RMSE keseluruhan.")
    if (dl["rmse_semua"] < sn["rmse_semua"]
            and dl["rasio_atenuasi_ekstrem"] < sn["rasio_atenuasi_ekstrem"]):
        print("\n  TEMUAN UTAMA: DLinear menurunkan RMSE keseluruhan di bawah seasonal")
        print("  naive, TETAPI rasio atenuasinya pada episode ekstrem lebih rendah.")
        print("  Model menurunkan galat rata-rata sambil membuang informasi amplitudo")
        print("  yang sudah tersedia di data (seasonal naive cukup menyalin hari kemarin).")

    # pembanding dari metrik yang sudah ditetapkan, supaya temuan di atas tidak
    # dibaca melebihi apa yang ditunjukkan datanya
    print("\n  PEMBANDING UNTUK MEMBACA TEMUAN DI ATAS:")
    ra_e, ra_n = dl["rasio_atenuasi_ekstrem"], dl["rasio_atenuasi_normal_tingkat_jendela"]
    print(f"  Rasio atenuasi DLinear: ekstrem {ra_e:.3f} (episode), "
          f"{dl['rasio_atenuasi_ekstrem_tingkat_jendela']:.3f} (jendela); normal {ra_n:.3f} (jendela)")
    if ra_n <= dl["rasio_atenuasi_ekstrem_tingkat_jendela"]:
        print("  -> DLinear meratakan jendela normal SAMA ATAU LEBIH kuat daripada jendela")
        print("     ekstrem. Perataan tidak khusus pada kejadian ekstrem.")
    else:
        print("  -> DLinear meratakan jendela ekstrem lebih kuat daripada jendela normal.")
    for nama, b in (("persistence", ps), ("seasonal naive", sn), ("DLinear", dl)):
        print(f"  RMSE ekstrem / RMSE normal {nama:<15}: {b['rmse_ekstrem'] / b['rmse_normal']:.2f}"
              f"   % SSE dari ekstrem: {b['persen_SSE_dari_ekstrem']:.1f}")
    terbaik = t["rmse_ekstrem"].idxmin()
    print(f"  RMSE ekstrem terendah dari ketiga model: {terbaik}")


def tahap_3(ctx, seed):
    print("\n" + "=" * 92)
    print(f"TAHAP 3 — DLINEAR, LOSS MSE (seed {seed})")
    print("=" * 92)
    pred, info = latih_dlinear(ctx, seed)
    ctx["prediksi"]["dlinear"] = pred
    ctx["info_latih"] = info

    print("\nMETRIK BAGIAN UJI, KETIGA MODEL")
    evaluasi(ctx, ctx["prediksi"])
    bandingkan(ctx["metrik"])
    print("\n  Jendela per bagian: " + ", ".join(
        f"{b} {len(ctx['jendela'][b][0]):,}" for b in PORSI))
    return ctx


def main():
    p = argparse.ArgumentParser(description="Eksperimen atenuasi ekstrem oleh MSE")
    p.add_argument("folder", nargs="?", default="hasil", help="folder hasil")
    p.add_argument("--mentah", default=os.path.join("data_mentah", DATASET),
                   help="folder data mentah, untuk pemeriksaan status QAQC")
    p.add_argument("--tahap", type=int, choices=[1, 2, 3], default=3,
                   help="jalankan sampai tahap ini")
    p.add_argument("--seed", type=int, default=SEED, help="seed acak")
    args = p.parse_args()

    np.random.seed(args.seed)
    print(f">> Seed: {args.seed}")
    ctx = tahap_1(args.folder, args.mentah)
    if args.tahap >= 2:
        if not ctx["lolos"]:
            sys.exit("\nSyarat musim panas tidak terpenuhi. Tahap 2 tidak dijalankan.")
        tahap_2(ctx)
    if args.tahap >= 3:
        tahap_3(ctx, args.seed)


if __name__ == "__main__":
    main()
