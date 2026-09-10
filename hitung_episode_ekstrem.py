#!/usr/bin/env python3
"""
hitung_episode_ekstrem.py
=========================
Menghitung jumlah dan karakteristik EPISODE EKSTREM oksigen terlarut (DO)
pada satu deret waktu kualitas air.

Tujuan: menguji kelayakan premis penelitian SEBELUM membangun model apa pun.
Pertanyaan yang dijawab skrip ini:
  1. Berapa banyak episode ekstrem yang benar-benar ada di dataset ini?
  2. Apakah jumlahnya cukup untuk evaluasi yang bermakna secara statistik?
  3. Apakah temuannya bertahan ketika definisi ambang digeser?
  4. Seberapa kuat siklus harian DO di lokasi ini?

Cara pakai:
    python hitung_episode_ekstrem.py data.csv --waktu DateTimeStamp --do DO_mgl
    python hitung_episode_ekstrem.py --demo      # jalankan dengan data sintetis

Keluaran: ringkasan di layar + file CSV berisi daftar episode + tabel sensitivitas.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# KONFIGURASI DEFAULT (silakan ubah sesuai dataset Anda)
# ----------------------------------------------------------------------------
FREKUENSI = "15min"        # resolusi target setelah resampling
AMBANG_MUTLAK = 3.0        # mg/L, ambang utama definisi D1
DURASI_MIN_MENIT = 60      # episode harus bertahan minimal sekian menit
JEDA_GABUNG_MENIT = 60     # dua episode berjarak < ini dianggap satu episode
LAJU_TURUN = 0.5           # mg/L per jam, ambang definisi D2
PERSENTIL = 1.0            # persentil terbawah, definisi D3 dan D4

SWEEP_AMBANG = [2.0, 2.5, 3.0, 3.5, 4.0]        # untuk analisis sensitivitas
SWEEP_DURASI = [30, 60, 120, 240]               # menit


# ----------------------------------------------------------------------------
# 1. PEMUATAN DAN PEMBERSIHAN
# ----------------------------------------------------------------------------
def muat_data(path, kol_waktu, kol_do, kol_lain=None):
    """Baca CSV, jadikan indeks waktu, sisakan kolom yang dipakai."""
    df = pd.read_csv(path)
    if kol_waktu not in df.columns:
        sys.exit(f"Kolom waktu '{kol_waktu}' tidak ada. Kolom tersedia: {list(df.columns)}")
    if kol_do not in df.columns:
        sys.exit(f"Kolom DO '{kol_do}' tidak ada. Kolom tersedia: {list(df.columns)}")

    df[kol_waktu] = pd.to_datetime(df[kol_waktu], errors="coerce")
    df = df.dropna(subset=[kol_waktu]).set_index(kol_waktu).sort_index()

    kolom = [kol_do] + [c for c in (kol_lain or []) if c in df.columns]
    df = df[kolom].apply(pd.to_numeric, errors="coerce")
    return df.rename(columns={kol_do: "DO"})


def rapikan(df, freq=FREKUENSI):
    """Samakan ke grid waktu teratur. TIDAK mengisi lubang besar."""
    df = df[~df.index.duplicated(keep="first")]
    return df.resample(freq).mean()


def laporan_kualitas(df, freq=FREKUENSI):
    """Cek kelengkapan data. Lubang besar merusak deteksi episode."""
    total = len(df)
    ada = int(df["DO"].notna().sum())
    langkah = pd.Timedelta(freq)
    rentang_hari = (df.index[-1] - df.index[0]).total_seconds() / 86400

    hilang = df["DO"].isna()
    grup = (hilang != hilang.shift()).cumsum()
    runs = hilang.groupby(grup).agg(["sum", "size"])
    lubang = runs[runs["sum"] > 0]["size"] * langkah.total_seconds() / 3600

    print("\n" + "=" * 72)
    print("LANGKAH 1 — KUALITAS DATA")
    print("=" * 72)
    print(f"  Rentang waktu      : {df.index[0]}  s/d  {df.index[-1]}")
    print(f"  Durasi             : {rentang_hari:,.0f} hari (~{rentang_hari/365:.1f} tahun)")
    print(f"  Titik pada grid    : {total:,}")
    print(f"  Titik terisi       : {ada:,}  ({100*ada/total:.1f}%)")
    if len(lubang):
        print(f"  Jumlah lubang      : {len(lubang):,}")
        print(f"  Lubang terpanjang  : {lubang.max():,.1f} jam")
        print(f"  Lubang > 24 jam    : {int((lubang > 24).sum()):,}")
    else:
        print("  Tidak ada data hilang.")
    if 100 * ada / total < 80:
        print("  PERINGATAN: kelengkapan < 80%. Pertimbangkan stasiun/periode lain.")
    return rentang_hari


# ----------------------------------------------------------------------------
# 2. MESIN EKSTRAKSI EPISODE
# ----------------------------------------------------------------------------
def ekstrak_episode(mask, seri, freq=FREKUENSI,
                    durasi_min_menit=DURASI_MIN_MENIT,
                    jeda_gabung_menit=JEDA_GABUNG_MENIT):
    """
    Ubah mask boolean (True = kondisi ekstrem) menjadi daftar EPISODE.

    Ini inti metodologisnya: unit analisis adalah episode, bukan titik.
    Memprediksi satu titik rendah tidak sama nilainya dengan memprediksi
    seluruh episode krisis.
    """
    langkah_menit = pd.Timedelta(freq).total_seconds() / 60
    m = mask.fillna(False).to_numpy()
    if not m.any():
        return pd.DataFrame(columns=["mulai", "selesai", "durasi_jam",
                                     "nilai_min", "nilai_rata", "jam_mulai"])

    # cari batas blok True
    tepi = np.diff(np.concatenate(([0], m.view(np.int8), [0])))
    awal = np.where(tepi == 1)[0]
    akhir = np.where(tepi == -1)[0]  # eksklusif

    # gabungkan blok yang berdekatan
    jeda_langkah = int(round(jeda_gabung_menit / langkah_menit))
    gabung_a, gabung_b = [awal[0]], [akhir[0]]
    for a, b in zip(awal[1:], akhir[1:]):
        if a - gabung_b[-1] <= jeda_langkah:
            gabung_b[-1] = b
        else:
            gabung_a.append(a)
            gabung_b.append(b)

    baris = []
    min_langkah = max(1, int(round(durasi_min_menit / langkah_menit)))
    for a, b in zip(gabung_a, gabung_b):
        if b - a < min_langkah:
            continue  # terlalu pendek, buang
        potongan = seri.iloc[a:b]
        baris.append({
            "mulai": seri.index[a],
            "selesai": seri.index[b - 1],
            "durasi_jam": (b - a) * langkah_menit / 60,
            "nilai_min": potongan.min(),
            "nilai_rata": potongan.mean(),
            "jam_mulai": seri.index[a].hour,
        })
    return pd.DataFrame(baris)


# ----------------------------------------------------------------------------
# 3. EMPAT DEFINISI EKSTREM
# ----------------------------------------------------------------------------
def d1_ambang_durasi(df, ambang=AMBANG_MUTLAK, durasi=DURASI_MIN_MENIT):
    """D1: DO di bawah ambang mutlak, bertahan minimal <durasi> menit.
    Berakar pada standar budidaya. Paling mudah dipertahankan di ujian."""
    return ekstrak_episode(df["DO"] < ambang, df["DO"], durasi_min_menit=durasi)


def d2_laju_turun(df, laju=LAJU_TURUN, freq=FREKUENSI):
    """D2: DO turun lebih cepat dari <laju> mg/L per jam.
    Menangkap kondisi MENUJU krisis sebelum ambang terlampaui.
    Inilah yang relevan untuk peringatan dini."""
    per_jam = pd.Timedelta("1h") / pd.Timedelta(freq)
    delta = df["DO"].diff() * per_jam          # mg/L per jam
    return ekstrak_episode(delta < -laju, df["DO"], durasi_min_menit=30)


def d3_persentil(df, p=PERSENTIL, durasi=DURASI_MIN_MENIT):
    """D3: DO di bawah persentil ke-p dari distribusi historisnya sendiri.
    Bebas domain, bisa dipakai di dataset apa pun termasuk benchmark
    yang tidak ada hubungannya dengan air. Ini yang menopang klaim umum."""
    ambang = df["DO"].quantile(p / 100.0)
    ep = ekstrak_episode(df["DO"] < ambang, df["DO"], durasi_min_menit=durasi)
    return ep, ambang


def baseline_harian_musiman(df, freq=FREKUENSI):
    """Pisahkan DO menjadi komponen musiman + harian + residual.

    Dipakai oleh D4. Sekaligus menghasilkan bukti kuantitatif untuk argumen
    bahwa 'ekstrem' bersifat KONTEKSTUAL: DO 3.5 mg/L jam 5 pagi itu normal,
    nilai yang sama jam 2 siang adalah tanda bahaya.
    """
    do = df["DO"]
    per_hari = int(pd.Timedelta("1D") / pd.Timedelta(freq))

    # komponen musiman: median bergerak 15 hari (tahan terhadap pencilan)
    musiman = do.rolling(15 * per_hari, center=True, min_periods=per_hari).median()
    tanpa_musim = do - musiman

    # komponen harian: rata-rata per (bulan, jam) agar amplitudo diurnal
    # boleh berbeda antar musim
    kunci = pd.MultiIndex.from_arrays([do.index.month, do.index.hour])
    harian_peta = tanpa_musim.groupby(kunci).transform("mean")

    residual = tanpa_musim - harian_peta
    return musiman, harian_peta, residual


def d4_kontekstual(df, p=PERSENTIL, durasi=DURASI_MIN_MENIT):
    """D4: residual (setelah musiman dan siklus harian dibuang) di bawah
    persentil ke-p. Ini definisi ekstrem yang SADAR KONTEKS."""
    _, _, residual = baseline_harian_musiman(df)
    ambang = residual.quantile(p / 100.0)
    ep = ekstrak_episode(residual < ambang, df["DO"], durasi_min_menit=durasi)
    return ep, ambang, residual


# ----------------------------------------------------------------------------
# 4. RINGKASAN DAN SENSITIVITAS
# ----------------------------------------------------------------------------
def ringkas(nama, ep, tahun):
    if len(ep) == 0:
        return {"definisi": nama, "n_episode": 0, "per_tahun": 0.0,
                "durasi_median_jam": np.nan, "durasi_maks_jam": np.nan,
                "nilai_min": np.nan}
    return {
        "definisi": nama,
        "n_episode": len(ep),
        "per_tahun": len(ep) / tahun,
        "durasi_median_jam": ep["durasi_jam"].median(),
        "durasi_maks_jam": ep["durasi_jam"].max(),
        "nilai_min": ep["nilai_min"].min(),
    }


def vonis(n):
    """Aturan praktis kelayakan statistik."""
    if n >= 100:
        return "MEMADAI  - cukup untuk metrik terkondisi & uji signifikansi"
    if n >= 30:
        return "MARJINAL - bisa dipakai, laporkan selang kepercayaan"
    if n >= 10:
        return "LEMAH    - hanya untuk ilustrasi, jangan jadi klaim utama"
    return "TIDAK LAYAK - cari stasiun lain atau longgarkan definisi"


def sensitivitas(df, tahun):
    print("\n" + "=" * 72)
    print("LANGKAH 4 — SENSITIVITAS TERHADAP PILIHAN AMBANG (D1)")
    print("=" * 72)
    print("Ini yang mematikan pertanyaan 'kenapa ambangnya segitu?' di ujian.\n")
    baris = []
    for amb in SWEEP_AMBANG:
        for dur in SWEEP_DURASI:
            ep = d1_ambang_durasi(df, ambang=amb, durasi=dur)
            baris.append({"ambang_mgL": amb, "durasi_min_menit": dur,
                          "n_episode": len(ep),
                          "per_tahun": round(len(ep) / tahun, 1)})
    tabel = pd.DataFrame(baris)
    pivot = tabel.pivot(index="ambang_mgL", columns="durasi_min_menit",
                        values="n_episode")
    print("Jumlah episode menurut ambang (baris) dan durasi minimum (kolom):")
    print(pivot.to_string())
    return tabel


def profil_harian(df):
    print("\n" + "=" * 72)
    print("LANGKAH 5 — KEKUATAN SIKLUS HARIAN")
    print("=" * 72)
    per_jam = df["DO"].groupby(df.index.hour).mean()
    amplitudo = per_jam.max() - per_jam.min()
    print(f"  DO rata-rata tertinggi : {per_jam.max():.2f} mg/L (jam {per_jam.idxmax():02d})")
    print(f"  DO rata-rata terendah  : {per_jam.min():.2f} mg/L (jam {per_jam.idxmin():02d})")
    print(f"  Amplitudo diurnal      : {amplitudo:.2f} mg/L")
    print(f"  Simpangan baku total   : {df['DO'].std():.2f} mg/L")
    print(f"  Rasio amplitudo/SD     : {amplitudo/df['DO'].std():.2f}")
    if amplitudo / df["DO"].std() > 0.5:
        print("\n  Siklus harian KUAT. Ini bukti kuantitatif bahwa ambang tetap")
        print("  akan salah menilai: nilai yang sama bermakna berbeda per jam.")
        print("  Pakai angka ini di Bab I sebagai justifikasi definisi kontekstual.")
    return per_jam


# ----------------------------------------------------------------------------
# 5. DATA SINTETIS UNTUK UJI SKRIP
# ----------------------------------------------------------------------------
def buat_demo(tahun=3, seed=42):
    """Deret waktu DO tiruan: musiman + diurnal + hipoksia sesekali."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=tahun * 365 * 96, freq="15min")
    t = np.arange(len(idx))

    musiman = 7.0 + 1.8 * np.sin(2 * np.pi * t / (365 * 96) - 1.2)
    diurnal = 1.5 * np.sin(2 * np.pi * (t % 96) / 96 - 1.9)
    derau = rng.normal(0, 0.35, len(idx))
    do = musiman + diurnal + derau

    # sisipkan ~8 peristiwa hipoksia per tahun, lebih sering di musim panas
    for _ in range(8 * tahun):
        mulai = rng.integers(0, len(idx) - 400)
        panjang = int(rng.integers(24, 300))          # 6 sampai 75 jam
        dalam = rng.uniform(2.5, 5.0)
        bentuk = np.sin(np.linspace(0, np.pi, panjang))
        do[mulai:mulai + panjang] -= dalam * bentuk

    do = np.clip(do, 0.1, None)
    # buang sebagian data untuk meniru sensor mati
    do[rng.random(len(idx)) < 0.03] = np.nan
    mati = rng.integers(0, len(idx) - 900)
    do[mati:mati + 900] = np.nan

    return pd.DataFrame({"DO": do}, index=idx)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def analisis_lengkap(df, label="dataset", keluaran=".", dataset=None):
    """Jalankan seluruh enam langkah pada satu deret waktu DO yang sudah rapi.

    Dipanggil dari main() maupun dari jalankan_semua.py. Semua berkas keluaran
    ditulis ke subfolder di bawah <keluaran>, bukan ke direktori kerja.
    Kembaliannya tabel ringkasan empat definisi.
    """
    for sub in ("episode", "ringkasan", "sensitivitas"):
        os.makedirs(os.path.join(keluaran, sub), exist_ok=True)

    rentang_hari = laporan_kualitas(df)
    tahun = rentang_hari / 365.25

    print("\n" + "=" * 72)
    print("LANGKAH 2 — SEBARAN NILAI DO")
    print("=" * 72)
    for q in [0.1, 0.5, 1, 5, 25, 50, 75, 99]:
        print(f"  persentil {q:>5.1f} : {df['DO'].quantile(q/100):6.2f} mg/L")

    print("\n" + "=" * 72)
    print("LANGKAH 3 — JUMLAH EPISODE EKSTREM MENURUT EMPAT DEFINISI")
    print("=" * 72)

    hasil, simpan = [], {}

    ep1 = d1_ambang_durasi(df)
    hasil.append(ringkas(f"D1 ambang<{AMBANG_MUTLAK} & >={DURASI_MIN_MENIT}mnt", ep1, tahun))
    simpan["D1"] = ep1

    ep2 = d2_laju_turun(df)
    hasil.append(ringkas(f"D2 laju turun >{LAJU_TURUN} mg/L/jam", ep2, tahun))
    simpan["D2"] = ep2

    ep3, amb3 = d3_persentil(df)
    hasil.append(ringkas(f"D3 persentil {PERSENTIL}% (={amb3:.2f} mg/L)", ep3, tahun))
    simpan["D3"] = ep3

    ep4, amb4, _ = d4_kontekstual(df)
    hasil.append(ringkas(f"D4 residual kontekstual {PERSENTIL}%", ep4, tahun))
    simpan["D4"] = ep4

    tabel = pd.DataFrame(hasil)
    print(tabel.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n  VONIS KELAYAKAN:")
    for _, r in tabel.iterrows():
        print(f"    {r['definisi']:<44} n={int(r['n_episode']):>4}  {vonis(r['n_episode'])}")

    tab_sens = sensitivitas(df, tahun)
    profil_harian(df)

    # tumpang tindih antar definisi: apakah keempatnya menandai kejadian yang sama?
    print("\n" + "=" * 72)
    print("LANGKAH 6 — APAKAH KEEMPAT DEFINISI MENUNJUK KEJADIAN YANG SAMA?")
    print("=" * 72)
    for nama, ep in simpan.items():
        if len(ep):
            jam = ep["jam_mulai"].value_counts().sort_index()
            print(f"  {nama}: {len(ep):>4} episode, jam mulai paling sering "
                  f"= {jam.idxmax():02d}:00 ({jam.max()} kali)")

    # kolom penanda ikut ditulis ke disk supaya tiap berkas ringkasan berdiri
    # sendiri dan bisa langsung digabung lintas dataset tanpa ditambahi lagi
    for t in (tabel, tab_sens):
        t.insert(0, "label", label)
        if dataset is not None:
            t.insert(0, "dataset", dataset)
    tabel.to_csv(os.path.join(keluaran, "ringkasan", f"{label}.csv"), index=False)
    tab_sens.to_csv(os.path.join(keluaran, "sensitivitas", f"{label}.csv"), index=False)
    for nama, ep in simpan.items():
        if len(ep):
            ep.to_csv(os.path.join(keluaran, "episode", f"{label}_{nama}.csv"),
                      index=False)

    print("\n" + "=" * 72)
    print(f"SELESAI. Hasil ditulis ke '{keluaran}'.")
    print("=" * 72)
    return tabel


def main():
    p = argparse.ArgumentParser(description="Hitung episode ekstrem DO")
    p.add_argument("csv", nargs="?", help="berkas CSV data")
    p.add_argument("--waktu", default="DateTimeStamp", help="nama kolom waktu")
    p.add_argument("--do", default="DO_mgl", help="nama kolom DO")
    p.add_argument("--demo", action="store_true", help="pakai data sintetis")
    p.add_argument("--label", default="dataset", help="nama untuk berkas keluaran")
    p.add_argument("--keluaran", default="hasil", help="folder hasil")
    args = p.parse_args()

    if args.demo:
        print(">> MODE DEMO: data sintetis, bukan data nyata.")
        df = buat_demo()
    elif args.csv:
        df = rapikan(muat_data(args.csv, args.waktu, args.do))
    else:
        sys.exit("Beri nama berkas CSV, atau gunakan --demo.")

    analisis_lengkap(df, label=args.label, keluaran=args.keluaran)


if __name__ == "__main__":
    main()
