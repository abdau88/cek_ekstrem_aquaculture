#!/usr/bin/env python3
"""
siapkan_data_nerrs.py
=====================
Menggabungkan dan MEMBERSIHKAN data SWMP NERRS menjadi satu berkas CSV bersih
per stasiun, siap dipakai hitung_episode_ekstrem.py.

Kenapa langkah ini wajib dan tidak boleh dilewati:
  Berkas mentah NERRS memuat baris yang nilainya ADA tetapi sudah DITOLAK
  oleh QAQC (flag negatif). Nilai seperti itu sering berupa pembacaan rendah
  palsu akibat sensor kotor, biofouling, atau kalibrasi melenceng. Kalau ikut
  terbaca, skrip penghitung episode akan melaporkan episode ekstrem yang
  sebenarnya adalah kegagalan alat. Itu kesalahan fatal untuk disertasi.

DUA TATA LETAK yang dikenali otomatis:
  A. Satu berkas per stasiun per tahun, mis. aceeiwq2023.csv  (folder 288653)
  B. Satu berkas gabungan berkolom StationCode, mis. 23953.csv (folder 23953)

Kode flag NERRS (kolom F_*):
     0  lolos QAQC                      <- dipakai
     1  suspect / dicurigai             <- opsional, lihat --suspect
     2  dicadangkan
     3  data terhitung (koreksi barometrik)
     4  data historis pra-QAQC otomatis  <- dipakai
     5  data terkoreksi                  <- dipakai
    -1  parameter opsional tidak diukur  <- dibuang
    -2  data hilang                      <- dibuang
    -3  ditolak QAQC                     <- dibuang
    -4  di bawah rentang sensor          <- dibuang
    -5  di atas rentang sensor           <- dibuang

Biasanya dipanggil lewat jalankan_semua.py, tetapi bisa berdiri sendiri:
    python siapkan_data_nerrs.py data_mentah/23953 --keluaran hasil/23953/bersih
"""

import argparse
import glob
import os
import re
import sys

import pandas as pd

FREKUENSI = "15min"
# nama kanonis -> ejaan alternatif yang pernah muncul di berkas NERRS
PARAMETER = {
    "DO_mgl": ["DO_mgl"],
    "Temp": ["Temp"],
    "Sal": ["Sal"],
    "SpCond": ["SpCond"],
    "pH": ["pH"],
    "Turb": ["Turb"],
    "DO_Pct": ["DO_Pct", "DO_pct"],
    "Depth": ["Depth"],
}
FLAG_DITERIMA = {0, 4, 5}          # selalu dipakai
FLAG_SUSPECT = {1}                 # nasibnya ditentukan --suspect

POLA_FLAG = re.compile(r"<(-?\d+)>")
POLA_TAHUNAN = re.compile(r"^(?P<kode>[a-z]{5})wq(?P<tahun>\d{4})\.csv$", re.I)


def kode_flag(seri):
    """Ambil angka di dalam <> dari kolom flag. Contoh '<-3> [GSM] (CWD)' -> -3."""
    return pd.to_numeric(seri.astype(str).str.extract(POLA_FLAG, expand=False),
                         errors="coerce")


def petakan_kolom(kolom_tersedia):
    """Cocokkan nama kolom kanonis dengan ejaan yang benar-benar ada di berkas.

    Perlu karena tata letak A memakai 'DO_Pct' sedangkan tata letak B 'DO_pct'.
    Pencocokan dibuat tidak peka huruf besar-kecil.
    """
    lookup = {c.strip().lower(): c.strip() for c in kolom_tersedia}
    peta = {}
    for kanonis, alias in PARAMETER.items():
        for a in alias:
            if a.lower() in lookup:
                peta[kanonis] = lookup[a.lower()]
                break
    return peta


def bersihkan(df, peta, buang_suspect):
    """Terapkan flag QAQC: nilai yang tidak lolos diganti NaN. Ubah ke nama kanonis."""
    diterima = FLAG_DITERIMA if buang_suspect else FLAG_DITERIMA | FLAG_SUSPECT
    keluar, jejak = {}, {}
    for kanonis, asli in peta.items():
        nilai = pd.to_numeric(df[asli], errors="coerce")
        sebelum = int(nilai.notna().sum())
        kf = "F_" + asli
        if kf in df.columns:
            nilai = nilai.where(kode_flag(df[kf]).isin(diterima))
        keluar[kanonis] = nilai
        jejak[kanonis] = (sebelum, int(nilai.notna().sum()))
    return pd.DataFrame(keluar, index=df.index), jejak


def ke_grid(df):
    """Urutkan, buang duplikat waktu, samakan ke grid 15 menit."""
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df.resample(FREKUENSI).mean()


def statistik(kode, df, jejak, tujuan, n_sumber):
    do = df["DO_mgl"]
    total, ada = len(df), int(do.notna().sum())
    mentah, sisa = jejak.get("DO_mgl", (0, 0))
    return {
        "stasiun": kode,
        "n_sumber": n_sumber,
        "mulai": df.index[0].date(),
        "selesai": df.index[-1].date(),
        "tahun": round((df.index[-1] - df.index[0]).days / 365.25, 2),
        "titik_grid": total,
        "DO_terisi": ada,
        "kelengkapan_%": round(100 * ada / total, 1) if total else 0.0,
        "dibuang_QAQC": mentah - sisa,
        "DO_min": round(do.min(), 2) if ada else None,
        "DO_p1": round(do.quantile(0.01), 2) if ada else None,
        "DO_median": round(do.median(), 2) if ada else None,
        "jam_dibawah_3": round(int((do < 3.0).sum()) * 0.25, 1),
        "berkas": tujuan,
    }


def baca_kepala(path):
    """Ambil daftar nama kolom saja.

    Dibungkus try/except karena folder dataset ikut memuat berkas pendamping
    seperti sampling_stations.csv yang berpengkodean latin-1 dan bukan data
    deret waktu. Berkas semacam itu cukup dilewati, bukan menghentikan proses.
    """
    for enc in ("utf-8", "latin-1"):
        try:
            kepala = pd.read_csv(path, nrows=0, encoding=enc)
            return [c.strip() for c in kepala.columns]
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
    return []


def baca_mentah(path, kolom_ekstra=()):
    """Baca satu berkas NERRS, ambil hanya kolom yang dipakai."""
    kepala = pd.DataFrame(columns=baca_kepala(path))
    peta = petakan_kolom(kepala.columns)
    pakai = list(kolom_ekstra) + ["DateTimeStamp"]
    for asli in peta.values():
        pakai += [asli, "F_" + asli]
    pakai = [c for c in dict.fromkeys(pakai) if c in kepala.columns]

    df = pd.read_csv(path, usecols=pakai, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["DateTimeStamp"] = pd.to_datetime(df["DateTimeStamp"],
                                         format="%m/%d/%Y %H:%M", errors="coerce")
    return df.dropna(subset=["DateTimeStamp"]), peta


def proses_folder(folder, keluaran, buang_suspect, diam=False):
    """Deteksi tata letak, proses tiap stasiun, kembalikan daftar statistik."""
    berkas = sorted(glob.glob(os.path.join(folder, "*.csv")))
    tahunan = [b for b in berkas if POLA_TAHUNAN.match(os.path.basename(b))]
    os.makedirs(keluaran, exist_ok=True)

    def catat(pesan):
        if not diam:
            print(pesan, flush=True)

    if tahunan:
        catat(">> Tata letak A: satu berkas per stasiun per tahun.")
        kelompok = {}
        for b in tahunan:
            kelompok.setdefault(
                POLA_TAHUNAN.match(os.path.basename(b))["kode"].lower(), []).append(b)
        hasil = []
        for kode, daftar in sorted(kelompok.items()):
            catat(f"   memproses {kode} ({len(daftar)} berkas) ...")
            potongan, peta = [], {}
            for b in daftar:
                d, p = baca_mentah(b)
                peta.update(p)
                potongan.append(d)
            gabung = pd.concat(potongan, ignore_index=True).set_index("DateTimeStamp")
            bersih, jejak = bersihkan(gabung, petakan_kolom(gabung.columns), buang_suspect)
            bersih = ke_grid(bersih)
            tujuan = os.path.join(keluaran, f"{kode}.csv")
            bersih.to_csv(tujuan, index_label="DateTimeStamp")
            hasil.append(statistik(kode, bersih, jejak, tujuan, len(daftar)))
        return hasil

    # tata letak B: berkas gabungan yang punya kolom StationCode
    gabungan = []
    for b in berkas:
        kolom = baca_kepala(b)
        if "StationCode" in kolom and "DateTimeStamp" in kolom:
            gabungan.append(b)
    if not gabungan:
        return []

    catat(">> Tata letak B: berkas gabungan berkolom StationCode.")
    hasil = []
    for b in gabungan:
        catat(f"   membaca {os.path.basename(b)} ...")
        df, _ = baca_mentah(b, kolom_ekstra=["StationCode"])
        df["StationCode"] = df["StationCode"].astype(str).str.strip().str.lower()
        for kode, bagian in df.groupby("StationCode"):
            catat(f"   memproses {kode} ...")
            bagian = bagian.drop(columns=["StationCode"]).set_index("DateTimeStamp")
            bersih, jejak = bersihkan(bagian, petakan_kolom(bagian.columns), buang_suspect)
            bersih = ke_grid(bersih)
            tujuan = os.path.join(keluaran, f"{kode}.csv")
            bersih.to_csv(tujuan, index_label="DateTimeStamp")
            hasil.append(statistik(kode, bersih, jejak, tujuan, 1))
    return hasil


def main():
    p = argparse.ArgumentParser(description="Siapkan data SWMP NERRS")
    p.add_argument("folder", help="folder berisi berkas mentah NERRS")
    p.add_argument("--keluaran", default="bersih", help="folder hasil")
    p.add_argument("--suspect", choices=["buang", "pakai"], default="buang",
                   help="perlakuan flag <1> suspect (default: buang)")
    args = p.parse_args()

    if not os.path.isdir(args.folder):
        sys.exit(f"Folder '{args.folder}' tidak ada.")

    buang = args.suspect == "buang"
    print(f">> Flag suspect <1>: {'DIBUANG' if buang else 'DIPAKAI'}")
    hasil = proses_folder(args.folder, args.keluaran, buang)
    if not hasil:
        sys.exit(f"Tidak ada berkas NERRS yang dikenali di '{args.folder}'.")

    tabel = pd.DataFrame(hasil)
    print("\n" + "=" * 100)
    print("RINGKASAN PER STASIUN")
    print("=" * 100)
    print(tabel.drop(columns=["berkas"]).to_string(index=False))
    print(f"\n>> Berkas bersih ada di folder '{args.keluaran}'")


if __name__ == "__main__":
    main()
