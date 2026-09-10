#!/usr/bin/env python3
"""
profil_diurnal.py
=================
Membedah RAGAM oksigen terlarut menjadi tiga komponen dan mengukur kekuatan
siklus harian, termasuk per musim.

    python profil_diurnal.py hasil

Menelusuri semua subfolder bersih/ di bawah folder yang diberikan, jadi satu
perintah mencakup seluruh dataset sekaligus.

DUA ANGKA YANG DICARI
  1. Porsi ragam komponen harian.
     Kalau irama harian menjelaskan porsi besar dari ragam DO, artinya sebagian
     besar variasi adalah pola normal yang bisa diramalkan. Metrik yang
     memperlakukan seluruh variasi sama berat akan memberi nilai tinggi kepada
     model yang sekadar menghafal irama itu, tanpa pernah menangkap kejadian
     ekstrem. Ini dasar argumen Bab I.
  2. Amplitudo diurnal per musim.
     Di banyak estuari siklus harian jauh lebih kuat di musim panas, saat
     hipoksia juga paling sering terjadi. Kalau pola itu muncul, ada tautan
     langsung antara irama harian dan kejadian ekstrem, bukan sekadar dua
     fakta yang kebetulan berdampingan.

DEKOMPOSISI
  Memakai fungsi yang sama persis dengan definisi D4 di
  hitung_episode_ekstrem.py, supaya angka di sini konsisten dengan episode
  kontekstual yang sudah dihitung:

      DO = musiman + harian + residual

  musiman  median bergerak 15 hari
  harian   rata-rata per pasangan (bulan, jam), jadi amplitudo diurnal boleh
           berbeda antar musim
  residual sisanya

  Ketiganya tidak ortogonal sempurna, sehingga jumlah ragamnya tidak persis
  sama dengan ragam DO. Kolom `jumlah_per_total` melaporkan rasio itu apa
  adanya sebagai pemeriksaan kejujuran, bukan disembunyikan.

CATATAN BEDA ANGKA
  Amplitudo di sini dihitung pada deret yang sudah dibuang komponen
  musimannya, sedangkan `perbandingan_konteks.csv` menghitungnya pada DO
  mentah. Angkanya wajar berbeda sedikit. Yang di sini lebih tepat disebut
  amplitudo siklus harian murni.
"""

import argparse
import glob
import os
import sys

import pandas as pd

from hitung_episode_ekstrem import baseline_harian_musiman

MUSIM = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
         6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
URUT_MUSIM = ["DJF", "MAM", "JJA", "SON"]


def cari_berkas(akar):
    """Kumpulkan (dataset, stasiun, path) dari semua subfolder bersih/.

    Kalau folder yang diberikan ternyata langsung berisi CSV, folder itu
    sendiri yang dipakai, supaya skrip tetap berguna di luar struktur hasil/.
    """
    ketemu = []
    for p in sorted(glob.glob(os.path.join(akar, "*", "bersih", "*.csv"))):
        bagian = os.path.normpath(p).split(os.sep)
        ketemu.append((bagian[-3], os.path.splitext(bagian[-1])[0], p))
    if ketemu:
        return ketemu
    for p in sorted(glob.glob(os.path.join(akar, "*.csv"))):
        ketemu.append((os.path.basename(os.path.normpath(akar)),
                       os.path.splitext(os.path.basename(p))[0], p))
    return ketemu


def amplitudo(seri):
    """Selisih rata-rata jam tertinggi dan terendah. NaN kalau data tak cukup."""
    seri = seri.dropna()
    if len(seri) < 96:
        return None, None, None
    per_jam = seri.groupby(seri.index.hour).mean()
    if per_jam.isna().all():
        return None, None, None
    return (per_jam.max() - per_jam.min(), int(per_jam.idxmax()),
            int(per_jam.idxmin()))


def episode_per_musim(akar, dataset, stasiun):
    """Jumlah episode D1 per musim, kalau berkasnya ada."""
    p = os.path.join(akar, dataset, "episode", f"{stasiun}_D1.csv")
    if not os.path.exists(p):
        return {}
    ep = pd.read_csv(p, parse_dates=["mulai"])
    return ep["mulai"].dt.month.map(MUSIM).value_counts().to_dict()


def profil_satu(path, kol_waktu, kol_do):
    """Dekomposisi satu stasiun. Kembaliannya (baris ringkas, tabel musim)."""
    df = pd.read_csv(path, parse_dates=[kol_waktu], index_col=kol_waktu)
    if kol_do not in df.columns:
        return None, None
    df = df.rename(columns={kol_do: "DO"})
    if df["DO"].notna().sum() < 96 * 30:
        return None, None

    musiman, harian, residual = baseline_harian_musiman(df)

    # bandingkan ketiga komponen hanya pada waktu yang lengkap bertiga
    sah = musiman.notna() & harian.notna() & residual.notna()
    v_mus, v_har = musiman[sah].var(), harian[sah].var()
    v_res, v_tot = residual[sah].var(), df["DO"][sah].var()
    jumlah = v_mus + v_har + v_res

    amp, jam_maks, jam_min = amplitudo(harian + residual)
    baris = {
        "sd_DO": round(df["DO"].std(), 2),
        "ragam_musiman_%": round(100 * v_mus / jumlah, 1),
        "ragam_harian_%": round(100 * v_har / jumlah, 1),
        "ragam_residual_%": round(100 * v_res / jumlah, 1),
        "jumlah_per_total": round(jumlah / v_tot, 2),
        "amplitudo_diurnal": round(amp, 2) if amp else None,
        "jam_puncak": jam_maks,
        "jam_lembah": jam_min,
    }

    tanpa_musim = harian + residual
    musim_ke = tanpa_musim.index.month.map(MUSIM)
    baris_musim = []
    for m in URUT_MUSIM:
        a, jmaks, jmin = amplitudo(tanpa_musim[musim_ke == m])
        baris_musim.append({
            "musim": m,
            "amplitudo_diurnal": round(a, 2) if a else None,
            "jam_puncak": jmaks,
            "jam_lembah": jmin,
        })
    return baris, pd.DataFrame(baris_musim)


def main():
    p = argparse.ArgumentParser(description="Profil siklus harian DO")
    p.add_argument("folder", nargs="?", default="hasil",
                   help="folder hasil, atau folder berisi CSV bersih")
    p.add_argument("--waktu", default="DateTimeStamp", help="nama kolom waktu")
    p.add_argument("--do", default="DO_mgl", help="nama kolom DO")
    p.add_argument("--keluaran", default=None,
                   help="folder penyimpanan tabel (default: sama dengan folder)")
    args = p.parse_args()

    berkas = cari_berkas(args.folder)
    if not berkas:
        sys.exit(f"Tidak ada CSV bersih yang ditemukan di '{args.folder}'.")
    keluaran = args.keluaran or args.folder
    os.makedirs(keluaran, exist_ok=True)

    print(f">> {len(berkas)} stasiun ditemukan.\n")
    ringkas, per_musim = [], []
    for dataset, stasiun, path in berkas:
        print(f"   memproses {dataset}/{stasiun} ...", flush=True)
        baris, tabel_musim = profil_satu(path, args.waktu, args.do)
        if baris is None:
            print(f"      dilewati, data tidak memadai.")
            continue
        ringkas.append({"dataset": dataset, "stasiun": stasiun, **baris})

        n_ep = episode_per_musim(args.folder, dataset, stasiun)
        tabel_musim.insert(0, "stasiun", stasiun)
        tabel_musim.insert(0, "dataset", dataset)
        tabel_musim["episode_D1"] = tabel_musim["musim"].map(n_ep).fillna(0).astype(int)
        per_musim.append(tabel_musim)

    if not ringkas:
        sys.exit("Tidak ada stasiun yang bisa diproses.")

    t_ringkas = pd.DataFrame(ringkas)
    t_musim = pd.concat(per_musim, ignore_index=True)
    t_ringkas.to_csv(os.path.join(keluaran, "perbandingan_diurnal.csv"), index=False)
    t_musim.to_csv(os.path.join(keluaran, "perbandingan_diurnal_musim.csv"),
                   index=False)

    print("\n" + "=" * 92)
    print("DEKOMPOSISI RAGAM DO")
    print("=" * 92)
    print(t_ringkas.to_string(index=False))
    print("\n  jumlah_per_total mendekati 1 berarti ketiga komponen hampir tidak")
    print("  saling tumpang tindih, sehingga persentase di atas layak dibaca.")

    print("\n" + "=" * 92)
    print("AMPLITUDO SIKLUS HARIAN PER MUSIM, BERDAMPINGAN DENGAN EPISODE EKSTREM")
    print("=" * 92)
    lebar = t_musim.pivot_table(index=["dataset", "stasiun"], columns="musim",
                                values=["amplitudo_diurnal", "episode_D1"])
    lebar = lebar.reindex(columns=URUT_MUSIM, level=1)
    print(lebar.to_string())

    print(f"\n>> Tabel disimpan:")
    print(f"     {keluaran}/perbandingan_diurnal.csv")
    print(f"     {keluaran}/perbandingan_diurnal_musim.csv")


if __name__ == "__main__":
    main()
