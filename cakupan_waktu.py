#!/usr/bin/env python3
"""
cakupan_waktu.py
================
Mengukur berapa persen WAKTU yang dihabiskan stasiun di dalam episode ekstrem,
untuk definisi D1, D3, dan D4, sepanjang tahun dan per musim (JJA, DJF).

    python cakupan_waktu.py hasil

Dijalankan setelah jalankan_semua.py selesai. Membaca bersih/ dan episode/
dari setiap dataset di bawah folder yang diberikan.

KENAPA ANGKA INI PERLU
  Jumlah episode saja tidak menunjukkan seberapa langka kejadian ekstrem.
  Kalau satu stasiun menghabiskan belasan persen waktunya di bawah ambang,
  "ekstrem" di sana adalah keadaan biasa, dan model yang selalu meramalkan
  DO rendah akan tampak bagus. Persentase per musim menunjukkan apakah
  kejadian itu menumpuk di musim panas.

PENYEBUT: TITIK VALID, BUKAN GRID
  total_jam_valid = jumlah titik DO tidak-NaN * 0.25. Kalau seluruh grid
  waktu yang dipakai, lubang data ikut jadi penyebut dan persentasenya
  mengecil tanpa ada perubahan apa pun di lapangan.

PEMBILANG
  jam_dalam_episode = jumlah kolom durasi_jam dari berkas episode. Untuk
  persentase musiman, setiap episode dipetakan ke grid 15 menit dari mulai
  sampai selesai, lalu dihitung langkah yang jatuh di bulan musim tersebut.
  Dengan begitu episode yang melintasi pergantian bulan terbagi dengan benar.

CATATAN KEJUJURAN
  ekstrak_episode menggabungkan dua blok yang berjarak <= 60 menit, sehingga
  durasi_jam bisa memuat langkah yang DO-nya NaN. Langkah itu masuk pembilang
  tetapi tidak masuk penyebut. Kolom `jam_episode_NaN` melaporkan besarnya apa
  adanya supaya bisa dinilai apakah berpengaruh.

  Berkas episode tidak ditulis kalau jumlah episodenya nol (mis. D3 pada
  gndbhwq, lihat README). Nol itu dikonfirmasi lewat ringkasan/<stasiun>.csv;
  kalau tidak bisa dikonfirmasi, barisnya dikosongkan, bukan diisi nol.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from profil_diurnal import MUSIM, cari_berkas

DEFINISI = ["D1", "D3", "D4"]
JAM_PER_LANGKAH = 0.25
MUSIM_DILAPORKAN = ["JJA", "DJF"]


def n_episode_tercatat(akar, dataset, stasiun):
    """Jumlah episode per definisi menurut ringkasan/, untuk konfirmasi nol."""
    p = os.path.join(akar, dataset, "ringkasan", f"{stasiun}.csv")
    if not os.path.exists(p):
        return {}
    r = pd.read_csv(p, dtype={"dataset": str, "label": str})
    return dict(zip(r["definisi"].str[:2], r["n_episode"]))


def tandai_episode(indeks, ep):
    """Mask boolean pada grid: True untuk langkah yang berada di dalam episode.

    Kolom selesai di berkas episode bersifat inklusif (indeks[b - 1]).
    """
    mask = np.zeros(len(indeks), dtype=bool)
    if len(ep) == 0:
        return mask
    a = indeks.searchsorted(ep["mulai"].to_numpy(), side="left")
    b = indeks.searchsorted(ep["selesai"].to_numpy(), side="right")
    for i, j in zip(a, b):
        mask[i:j] = True
    return mask


def persen(pembilang, penyebut):
    return round(100 * pembilang / penyebut, 2) if penyebut else None


def cakupan_satu(akar, dataset, stasiun, path, kol_waktu, kol_do):
    """Satu baris per definisi untuk satu stasiun."""
    df = pd.read_csv(path, parse_dates=[kol_waktu], index_col=kol_waktu,
                     usecols=[kol_waktu, kol_do])
    valid = df[kol_do].notna().to_numpy()
    musim_ke = df.index.month.map(MUSIM).to_numpy()
    tercatat = n_episode_tercatat(akar, dataset, stasiun)

    baris = []
    for d in DEFINISI:
        hasil = {"dataset": dataset, "stasiun": stasiun, "definisi": d}
        p = os.path.join(akar, dataset, "episode", f"{stasiun}_{d}.csv")
        if os.path.exists(p):
            ep = pd.read_csv(p, parse_dates=["mulai", "selesai"])
        elif tercatat.get(d) == 0:
            ep = pd.DataFrame(columns=["mulai", "selesai", "durasi_jam"])
        else:
            print(f"      {d}: berkas episode tidak ada dan nol tidak "
                  f"terkonfirmasi, dikosongkan.")
            baris.append(hasil)
            continue

        dalam = tandai_episode(df.index, ep)
        jam_episode = float(ep["durasi_jam"].sum())
        if abs(dalam.sum() * JAM_PER_LANGKAH - jam_episode) > 1e-6:
            print(f"      PERINGATAN {d}: episode tidak cocok dengan grid bersih/ "
                  f"({dalam.sum() * JAM_PER_LANGKAH:.2f} vs {jam_episode:.2f} jam). "
                  f"Jalankan ulang pipeline dengan --ulang.")

        jam_valid = valid.sum() * JAM_PER_LANGKAH
        hasil.update({
            "n_episode": len(ep),
            "total_jam_valid": jam_valid,
            "jam_dalam_episode": jam_episode,
            "persen_waktu_episode": persen(jam_episode, jam_valid),
        })
        for m in MUSIM_DILAPORKAN:
            di_musim = musim_ke == m
            hasil[f"persen_waktu_{m}"] = persen((dalam & di_musim).sum(),
                                                (valid & di_musim).sum())
        hasil["jam_episode_NaN"] = (dalam & ~valid).sum() * JAM_PER_LANGKAH
        baris.append(hasil)
    return baris


def main():
    p = argparse.ArgumentParser(description="Cakupan waktu episode ekstrem DO")
    p.add_argument("folder", nargs="?", default="hasil", help="folder hasil")
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
    semua = []
    for dataset, stasiun, path in berkas:
        print(f"   memproses {dataset}/{stasiun} ...", flush=True)
        semua += cakupan_satu(args.folder, dataset, stasiun, path,
                              args.waktu, args.do)

    tabel = pd.DataFrame(semua)
    tabel.to_csv(os.path.join(keluaran, "perbandingan_cakupan.csv"), index=False)

    kolom_persen = ["persen_waktu_episode"] + [f"persen_waktu_{m}"
                                               for m in MUSIM_DILAPORKAN]
    print("\n" + "=" * 110)
    print("PERSEN WAKTU VALID YANG BERADA DI DALAM EPISODE EKSTREM")
    print("=" * 110)
    for d in DEFINISI:
        bagian = tabel[tabel["definisi"] == d].drop(columns=["definisi"])
        print(f"\n{d}")
        print(bagian.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n" + "=" * 110)
    print("MEDIAN LINTAS STASIUN")
    print("=" * 110)
    median = tabel.groupby("definisi")[kolom_persen].median()
    print(median.to_string(float_format=lambda x: f"{x:.2f}"))

    nan_maks = (tabel["jam_episode_NaN"] / tabel["jam_dalam_episode"]).max()
    print(f"\n  jam_episode_NaN: porsi terbesar {100 * nan_maks:.2f}% dari "
          f"jam_dalam_episode.")
    print("  Kalau porsi ini kecil, perbedaan penyebut valid dan pembilang")
    print("  berbasis durasi_jam tidak mengubah kesimpulan.")

    print(f"\n>> Tabel disimpan:")
    print(f"     {keluaran}/perbandingan_cakupan.csv")


if __name__ == "__main__":
    main()
