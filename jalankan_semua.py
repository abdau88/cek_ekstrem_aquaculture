#!/usr/bin/env python3
"""
jalankan_semua.py
=================
SATU PERINTAH untuk seluruh pipeline, dari berkas mentah NERRS sampai tabel
perbandingan antar reserve.

    python jalankan_semua.py

Yang dikerjakan, untuk setiap folder dataset di dalam data_mentah/:
    1. deteksi tata letak berkas, gabungkan, terapkan flag QAQC   -> bersih/
    2. hitung episode ekstrem empat definisi per stasiun          -> episode/
    3. tulis ringkasan dan tabel sensitivitas per stasiun         -> ringkasan/, sensitivitas/
    4. simpan catatan layar lengkap tiap stasiun                  -> log/
    5. susun tabel perbandingan lintas dataset                    -> hasil/

MENAMBAH DATASET BARU nanti: cukup taruh foldernya di data_mentah/ lalu
jalankan perintah yang sama. Tidak ada yang perlu disunting di skrip ini.
Dataset yang hasilnya sudah lengkap akan dilewati, kecuali diberi --ulang.

Struktur folder yang dihasilkan:
    data_mentah/<dataset>/          berkas mentah apa adanya
    hasil/<dataset>/bersih/         satu CSV per stasiun, sudah lolos QAQC
    hasil/<dataset>/episode/        daftar episode per stasiun per definisi
    hasil/<dataset>/ringkasan/      empat definisi, jumlah dan durasi episode
    hasil/<dataset>/sensitivitas/   jumlah episode menurut ambang dan durasi
    hasil/<dataset>/log/            catatan layar lengkap
    hasil/perbandingan_stasiun.csv  satu baris per stasiun, semua dataset
    hasil/perbandingan_episode.csv  satu baris per stasiun per definisi
"""

import argparse
import contextlib
import io
import os
import sys
import traceback

import pandas as pd

from hitung_episode_ekstrem import analisis_lengkap
from siapkan_data_nerrs import proses_folder

DATA_MENTAH = "data_mentah"
HASIL = "hasil"
SUB_HASIL = ("bersih", "episode", "ringkasan", "sensitivitas", "log")


def daftar_dataset(akar):
    """Setiap subfolder data_mentah/ dianggap satu dataset, kecuali arsip/."""
    if not os.path.isdir(akar):
        sys.exit(f"Folder '{akar}' tidak ada. Taruh data mentah di sana dulu.")
    return sorted(
        d for d in os.listdir(akar)
        if os.path.isdir(os.path.join(akar, d)) and not d.startswith((".", "_"))
        and d != "arsip"
    )


def sudah_selesai(dir_hasil):
    """Dataset dilewati kalau folder ringkasannya sudah terisi."""
    r = os.path.join(dir_hasil, "ringkasan")
    return os.path.isdir(r) and any(f.endswith(".csv") for f in os.listdir(r))


def analisis_satu_stasiun(berkas, label, dir_hasil, dataset):
    """Jalankan enam langkah untuk satu stasiun. Layar ditangkap ke berkas log.

    Kembaliannya tabel ringkasan, atau None kalau stasiun ini gagal diproses.
    Kegagalan satu stasiun tidak boleh menghentikan dataset lainnya.
    """
    from hitung_episode_ekstrem import muat_data, rapikan

    dir_log = os.path.join(dir_hasil, "log")
    os.makedirs(dir_log, exist_ok=True)
    tangkap = io.StringIO()
    try:
        with contextlib.redirect_stdout(tangkap):
            df = rapikan(muat_data(berkas, "DateTimeStamp", "DO_mgl"))
            tabel = analisis_lengkap(df, label=label, keluaran=dir_hasil,
                                     dataset=dataset)
    except Exception:
        tangkap.write("\nGAGAL:\n" + traceback.format_exc())
        tabel = None
    with open(os.path.join(dir_log, f"{label}.txt"), "w", encoding="utf-8") as f:
        f.write(tangkap.getvalue())
    return tabel


def proses_dataset(nama, buang_suspect, dir_akar=DATA_MENTAH, dir_hasil_akar=HASIL):
    dir_mentah = os.path.join(dir_akar, nama)
    dir_hasil = os.path.join(dir_hasil_akar, nama)
    for sub in SUB_HASIL:
        os.makedirs(os.path.join(dir_hasil, sub), exist_ok=True)

    print(f"\n{'=' * 78}\nDATASET: {nama}\n{'=' * 78}")
    stat = proses_folder(dir_mentah, os.path.join(dir_hasil, "bersih"), buang_suspect)
    if not stat:
        print(f"   tidak ada berkas NERRS yang dikenali, dilewati.")
        return None, None

    tabel_stasiun = pd.DataFrame(stat)
    tabel_stasiun.insert(0, "dataset", nama)
    tabel_stasiun.to_csv(os.path.join(dir_hasil, "ringkasan_stasiun.csv"), index=False)
    print("\n" + tabel_stasiun.drop(columns=["berkas"]).to_string(index=False))

    print("\n   menghitung episode ekstrem ...")
    ringkasan = []
    for r in stat:
        label = r["stasiun"]
        tabel = analisis_satu_stasiun(r["berkas"], label, dir_hasil, nama)
        if tabel is None:
            print(f"      {label}: GAGAL, lihat log/{label}.txt")
            continue
        ringkasan.append(tabel)
        n = dict(zip(tabel["definisi"].str[:2], tabel["n_episode"]))
        print(f"      {label}: " + "  ".join(f"{k}={v}" for k, v in n.items()))

    gabung = pd.concat(ringkasan, ignore_index=True) if ringkasan else None
    return tabel_stasiun, gabung


def konteks_stasiun(dir_hasil_akar, daftar):
    """Susun bukti bahwa 'ekstrem' bersifat kontekstual, bukan nilai mutlak.

    Dua tabel ringkas, cukup kecil untuk dibawa ke diskusi tanpa data mentah:
      perbandingan_konteks.csv  satu baris per stasiun: korelasi DO terhadap
                                suhu, salinitas, kedalaman, plus amplitudo
                                siklus harian dibanding simpangan baku total
      perbandingan_bulanan.csv  DO rata-rata dan jumlah episode D1 per bulan

    Angka inilah yang menopang argumen bahwa ambang tetap salah menilai:
    kalau DO dikuasai suhu dan musim, satu ambang untuk sepanjang tahun akan
    menandai musim, bukan menandai kejadian ekstrem.
    """
    konteks, bulanan = [], []
    for dataset, label, berkas in daftar:
        # nama dataset kerap berupa angka (mis. 23953) dan terbaca sebagai int
        # ketika tabel dimuat ulang dari CSV, jadi dipaksa jadi teks di sini
        dataset, label = str(dataset), str(label)
        df = pd.read_csv(berkas, parse_dates=["DateTimeStamp"],
                         index_col="DateTimeStamp")
        if "DO_mgl" not in df.columns or df["DO_mgl"].notna().sum() == 0:
            continue
        do = df["DO_mgl"]

        baris = {"dataset": dataset, "stasiun": label}
        for pemicu in ("Temp", "Sal", "Depth"):
            baris[f"korelasi_{pemicu}"] = (
                round(do.corr(df[pemicu]), 3) if pemicu in df.columns else None)

        per_jam = do.groupby(do.index.hour).mean()
        amplitudo, sd = per_jam.max() - per_jam.min(), do.std()
        baris["amplitudo_diurnal"] = round(amplitudo, 2)
        baris["simpangan_baku"] = round(sd, 2)
        baris["rasio_amplitudo_sd"] = round(amplitudo / sd, 2) if sd else None
        konteks.append(baris)

        ep_path = os.path.join(dir_hasil_akar, dataset, "episode", f"{label}_D1.csv")
        per_bulan = pd.Series(dtype=int)
        if os.path.exists(ep_path):
            ep = pd.read_csv(ep_path, parse_dates=["mulai"])
            per_bulan = ep["mulai"].dt.month.value_counts()
        rata_bulan = do.groupby(do.index.month).mean()
        for bulan in range(1, 13):
            bulanan.append({
                "dataset": dataset, "stasiun": label, "bulan": bulan,
                "DO_rata": round(rata_bulan.get(bulan, float("nan")), 2),
                "episode_D1": int(per_bulan.get(bulan, 0)),
            })

    if not konteks:
        return
    t_konteks = pd.DataFrame(konteks)
    t_konteks.to_csv(os.path.join(dir_hasil_akar, "perbandingan_konteks.csv"),
                     index=False)
    pd.DataFrame(bulanan).to_csv(
        os.path.join(dir_hasil_akar, "perbandingan_bulanan.csv"), index=False)

    print("\n" + "=" * 78)
    print("KONTEKS: APA YANG SEBENARNYA MENGGERAKKAN DO")
    print("=" * 78)
    print(t_konteks.to_string(index=False))


def main():
    p = argparse.ArgumentParser(
        description="Jalankan seluruh pipeline untuk semua dataset di data_mentah/")
    p.add_argument("dataset", nargs="*",
                   help="nama folder dataset tertentu; kosong berarti semua")
    p.add_argument("--data", default=DATA_MENTAH, help="folder data mentah")
    p.add_argument("--keluaran", default=HASIL, help="folder hasil")
    p.add_argument("--suspect", choices=["buang", "pakai"], default="buang",
                   help="perlakuan flag <1> suspect (default: buang)")
    p.add_argument("--ulang", action="store_true",
                   help="proses ulang dataset yang hasilnya sudah ada")
    args = p.parse_args()

    semua = daftar_dataset(args.data)
    pilih = args.dataset or semua
    tak_dikenal = [d for d in pilih if d not in semua]
    if tak_dikenal:
        sys.exit(f"Dataset tidak ada di '{args.data}': {', '.join(tak_dikenal)}\n"
                 f"Yang tersedia: {', '.join(semua)}")

    buang = args.suspect == "buang"
    print(f">> Dataset ditemukan : {', '.join(semua)}")
    print(f">> Akan diproses     : {', '.join(pilih)}")
    print(f">> Flag suspect <1>  : {'DIBUANG' if buang else 'DIPAKAI'}")

    stasiun_semua, episode_semua = [], []
    for nama in pilih:
        dir_hasil = os.path.join(args.keluaran, nama)
        if not args.ulang and sudah_selesai(dir_hasil):
            print(f"\n>> {nama} sudah ada hasilnya, dilewati. Pakai --ulang untuk memaksa.")
            # dataset dan stasiun dibaca sebagai teks, kalau tidak nama seperti
            # 23953 berubah jadi bilangan dan merusak penyusunan path berikutnya
            teks = {"dataset": str, "stasiun": str, "label": str}
            berkas_stasiun = os.path.join(dir_hasil, "ringkasan_stasiun.csv")
            if os.path.exists(berkas_stasiun):
                stasiun_semua.append(pd.read_csv(berkas_stasiun, dtype=teks))
            r = os.path.join(dir_hasil, "ringkasan")
            episode_semua += [pd.read_csv(os.path.join(r, f), dtype=teks)
                              for f in sorted(os.listdir(r)) if f.endswith(".csv")]
            continue
        st, ep = proses_dataset(nama, buang, args.data, args.keluaran)
        if st is not None:
            stasiun_semua.append(st)
        if ep is not None:
            episode_semua.append(ep)

    if not stasiun_semua:
        sys.exit("\nTidak ada yang berhasil diproses.")

    os.makedirs(args.keluaran, exist_ok=True)
    tabel_stasiun = pd.concat(stasiun_semua, ignore_index=True)
    tabel_stasiun.to_csv(os.path.join(args.keluaran, "perbandingan_stasiun.csv"),
                         index=False)

    print("\n" + "=" * 78)
    print("PERBANDINGAN SEMUA STASIUN")
    print("=" * 78)
    print(tabel_stasiun.drop(columns=["berkas"]).to_string(index=False))

    if episode_semua:
        tabel_ep = pd.concat(episode_semua, ignore_index=True)
        tabel_ep.to_csv(os.path.join(args.keluaran, "perbandingan_episode.csv"),
                        index=False)
        tabel_ep["def"] = tabel_ep["definisi"].str[:2]
        pivot = tabel_ep.pivot_table(index=["dataset", "label"], columns="def",
                                     values="n_episode", aggfunc="first")
        print("\n" + "=" * 78)
        print("JUMLAH EPISODE PER DEFINISI")
        print("=" * 78)
        print(pivot.to_string())

    konteks_stasiun(args.keluaran,
                    list(zip(tabel_stasiun["dataset"], tabel_stasiun["stasiun"],
                             tabel_stasiun["berkas"])))

    print(f"\n>> Semua hasil ada di folder '{args.keluaran}'.")
    print(">> Tabel gabungan, cukup empat berkas ini untuk dibawa berdiskusi:")
    for f in ("perbandingan_stasiun.csv", "perbandingan_episode.csv",
              "perbandingan_konteks.csv", "perbandingan_bulanan.csv"):
        print(f"     {args.keluaran}/{f}")


if __name__ == "__main__":
    main()
