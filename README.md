# cek_ekstrem — uji kelayakan premis episode ekstrem DO

Menghitung jumlah dan karakteristik episode ekstrem oksigen terlarut (DO) pada
data SWMP NERRS, untuk menguji apakah premis penelitian layak SEBELUM model
apa pun dibangun.

## Satu perintah untuk semuanya

```
python jalankan_semua.py
```

Memproses setiap folder dataset di dalam `data_mentah/`, dari berkas mentah
sampai tabel perbandingan antar reserve.

### Menambah dataset baru

Taruh folder unduhan CDMO di `data_mentah/`, lalu jalankan perintah yang sama.
Tidak ada yang perlu disunting. Dataset yang hasilnya sudah lengkap dilewati
otomatis.

Dua tata letak berkas CDMO dikenali sendiri:

| Tata letak | Bentuk | Contoh |
|---|---|---|
| A | satu berkas per stasiun per tahun | `288653/aceeiwq2023.csv` |
| B | satu berkas gabungan berkolom `StationCode` | `23953/23953.csv` |

### Pilihan lain

```
python jalankan_semua.py 23953              # satu dataset saja
python jalankan_semua.py --ulang            # paksa proses ulang
python jalankan_semua.py --suspect pakai    # ikutkan data ber-flag <1>
```

Jalankan `--suspect pakai` lalu bandingkan hasilnya. Penguji akan bertanya
apakah temuan Anda bergantung pada keputusan pembersihan data.

## Struktur folder

```
data_mentah/<dataset>/          berkas CDMO apa adanya, tidak pernah disunting
data_mentah/arsip/              berkas zip unduhan asli

hasil/<dataset>/bersih/         satu CSV per stasiun, sudah lolos QAQC
hasil/<dataset>/episode/        daftar episode per stasiun per definisi
hasil/<dataset>/ringkasan/      empat definisi: jumlah, durasi, nilai minimum
hasil/<dataset>/sensitivitas/   jumlah episode menurut ambang dan durasi
hasil/<dataset>/log/            catatan layar lengkap enam langkah
hasil/<dataset>/ringkasan_stasiun.csv

hasil/perbandingan_stasiun.csv  satu baris per stasiun: rentang, kelengkapan, sebaran DO
hasil/perbandingan_episode.csv  satu baris per stasiun per definisi
hasil/perbandingan_konteks.csv  korelasi DO terhadap suhu, salinitas, kedalaman
hasil/perbandingan_bulanan.csv  DO rata-rata dan jumlah episode D1 per bulan
hasil/perbandingan_diurnal.csv        dekomposisi ragam: musiman, harian, residual
hasil/perbandingan_diurnal_musim.csv  amplitudo harian per musim dan episode D1
```

Empat berkas `perbandingan_*.csv` di akar `hasil/` adalah intisari seluruh
pipeline, total di bawah 10 KB. Itu saja yang perlu dibawa ke diskusi. Isi
folder `bersih/` masing-masing 11 MB dan tidak perlu dibagikan.

Folder `hasil/` seluruhnya bisa dihapus dan dibangun ulang. Folder
`data_mentah/` jangan disentuh.

## Skrip

| Berkas | Peran |
|---|---|
| `jalankan_semua.py` | orkestrator, titik masuk utama |
| `siapkan_data_nerrs.py` | deteksi tata letak, gabung, terapkan flag QAQC |
| `hitung_episode_ekstrem.py` | enam langkah analisis untuk satu deret waktu |
| `profil_diurnal.py` | dekomposisi ragam dan amplitudo harian per musim |

Ketiga yang terakhir bisa dijalankan sendiri, lihat docstring masing-masing.
`profil_diurnal.py` dijalankan terpisah setelah pipeline utama selesai:

```
python profil_diurnal.py hasil
```

## Kenapa penyaringan QAQC wajib

Berkas mentah NERRS memuat baris yang nilainya ADA tetapi sudah DITOLAK oleh
QAQC. Nilai seperti itu sering berupa pembacaan rendah palsu akibat sensor
kotor, biofouling, atau kalibrasi melenceng. Tanpa penyaringan, skrip akan
melaporkan episode ekstrem yang sebenarnya kegagalan alat.

Nilai dipakai kalau flagnya 0, 4, atau 5. Flag 1 (suspect) dibuang secara
baku. Semua flag negatif dibuang.

## Empat definisi ekstrem

| Kode | Definisi |
|---|---|
| D1 | DO di bawah ambang mutlak 3,0 mg/L, bertahan minimal 60 menit |
| D2 | DO turun lebih cepat dari 0,5 mg/L per jam |
| D3 | DO di bawah persentil ke-1 dari distribusinya sendiri |
| D4 | residual di bawah persentil ke-1, setelah musiman dan siklus harian dibuang |

Ambang dan parameter lain ada di bagian KONFIGURASI DEFAULT pada
`hitung_episode_ekstrem.py`.

### Catatan D3

D3 memakai pertidaksamaan tegas terhadap persentil ke-1. Di stasiun yang
anoksik kronis, persentil ke-1 bisa jatuh tepat di 0,0 mg/L sehingga tidak ada
nilai yang lolos dan D3 menghasilkan nol episode. Ini terjadi pada `gndbhwq`,
di mana 1,6% pembacaan bernilai persis 0,0. Perlakuan kasus ini adalah
keputusan metodologis, bukan bug.
