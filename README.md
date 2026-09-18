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
hasil/perbandingan_cakupan.csv        persen waktu valid di dalam episode D1/D3/D4, tahunan, JJA, DJF
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
| `cakupan_waktu.py` | persen waktu valid yang berada di dalam episode ekstrem |
| `eksperimen_atenuasi.py` | uji hipotesis atenuasi ekstrem oleh model ber-loss MSE (acemc) |

Keempat yang terakhir bisa dijalankan sendiri, lihat docstring masing-masing.
`profil_diurnal.py` dan `cakupan_waktu.py` dijalankan terpisah setelah pipeline utama selesai:

```
python profil_diurnal.py hasil
python cakupan_waktu.py hasil
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

## Eksperimen atenuasi (`eksperimen_atenuasi.py`)

Menguji satu hipotesis pada stasiun `acemc`: model yang dilatih dengan MSE
meratakan kejadian ekstrem, sehingga RMSE keseluruhan yang baik tidak berarti
kinerja baik pada episode kritis. Bukan upaya mengejar kinerja terbaik.

```
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python eksperimen_atenuasi.py hasil --tahap 3
python eksperimen_atenuasi.py hasil --tahap 3 --seed 7
```

Dikerjakan bertahap. Tahap 1 statistik jendela, tahap 2 persistence dan
seasonal naive, tahap 3 DLinear, tahap 4 gambar dan tabel akhir
`hasil/atenuasi_acemc.csv`. Pakai `python -m pip` dari interpreter yang sama
dengan yang menjalankan skrip; di mesin ini `pip` di PATH milik Python lain.

DLinear (rata-rata bergerak kernel 25, satu lapis linear untuk tren dan satu
untuk sisa) dilatih dengan MSE tanpa pembobotan, di CPU. Hiperparameter
mengikuti bawaan repositori DLinear dan tidak disetel: batch 32, Adam lr 0,005
yang dibagi dua setiap epoch, maksimal 10 epoch, early stopping kesabaran 3
pada MSE validasi. Normalisasi mean/std hanya dari bagian latih. Dengan seed
yang sama hasilnya identik antar-run.

Rancangan: DO univariat, input 288 langkah (72 jam), horizon 96 langkah
(24 jam) langsung, pembagian kronologis 70/10/20 pada grid waktu, jendela
stride 1 yang harus utuh di dalam satu bagian, jendela ber-NaN dibuang.
Jendela ekstrem = horizon aktual beririsan dengan episode D1.

### Agregasi dua tingkat

Dengan stride 1, ribuan jendela ekstrem berasal dari puluhan episode dan tidak
independen. Rasio atenuasi dan galat amplitudo puncak diringkas sebagai median
per episode lalu median lintas episode, dengan n = jumlah episode. Median
tingkat jendela ikut dilaporkan sebagai pembanding. Horizon yang menyentuh
lebih dari satu episode ditetapkan ke episode yang memuat DO aktual terendah;
akibatnya sebagian episode tidak memiliki jendela sendiri dan tidak masuk n.
Pada bagian uji: 66 episode tersentuh horizon, 53 memiliki jendela (n = 53),
13 tidak terwakili karena berdekatan dengan episode yang lebih dalam. Satu
jendela sengaja tidak dihitung di beberapa episode, supaya n tetap saling
lepas.

### Status: tahap 3 selesai, tahap 4 belum

Catatan layar lengkap tahap 1-3 (seed 42) ada di
`hasil/288653/log/atenuasi_acemc_tahap3.txt`.

Hasil pokok pada bagian uji: DLinear RMSE 0,205 mg/L, seasonal naive 0,344,
persistence 0,635. Rasio atenuasi tingkat episode DLinear 0,908 dan seasonal
naive 1,037, sedangkan rasio DLinear pada jendela normal 0,845. Jadi DLinear
kehilangan sebagian amplitudo, tetapi perataannya TIDAK khusus pada kejadian
ekstrem, dan DLinear justru model dengan RMSE ekstrem terendah. Versi kuat
hipotesis belum didukung di stasiun ini.

Dua keputusan yang masih terbuka sebelum tahap 4 dikerjakan:

1. Pemilihan 3 episode terdalam untuk gambar. Usulan: diambil dari 53 episode
   yang terwakili, diurutkan menurut `nilai_min`, nilai kembar dipecah dengan
   durasi terpanjang, dan jendela yang digambar menempatkan minimum episode di
   tengah horizon. Belum diputuskan apakah episode yang bertetangga hari boleh
   diambil dua-duanya.
2. Rumusan kriteria "hipotesis terbukti" untuk paragraf interpretasi, sebaiknya
   ditetapkan sebelum gambar dibuat.

Keterbatasan metrik yang perlu ikut disebut di interpretasi: rentang maks-min
aktual memuat derau sensor dan kuantisasi 0,1 mg/L sedangkan prediksi DLinear
halus, sehingga sebagian "atenuasi" hanyalah model yang tidak meramalkan derau.
Efek ini lebih besar pada jendela normal yang amplitudonya kecil.

### Analisis sekunder: rincian status QAQC

Ditetapkan sebelum hasil DLinear terlihat. Untuk setiap model, RMSE
keseluruhan, RMSE ekstrem, dan rasio atenuasi tingkat episode dilaporkan
terpisah untuk final (2021-2024), `ProvisionalPlus=1`, dan `ProvisionalPlus=0`.
Jendela ekstrem ikut status episode pemiliknya (di langkah mulai), jendela
normal ikut langkah pertama horizonnya. Kelompok dengan kurang dari 10 episode
dilaporkan apa adanya dengan peringatan, tidak digabung. Kelompok final kosong
di bagian uji.

### Penyimpangan dan keputusan yang disengaja

- **Musim panas uji tidak penuh.** Bagian uji (10 Jul 2025 s/d 27 Ags 2026)
  tidak memuat Jun-Ags yang utuh. Musim panas 2026 tercakup 95,2% karena data
  acemc berakhir 27 Agustus 2026, dan diterima sebagai memenuhi syarat.
  Menggeser batas uji ke Juni 2025 ditolak: musim panas 2025 hanya terisi
  43,2%, jadi itu menukar musim panas yang baik dengan yang buruk sambil
  merusak validasi.
- **Validasi miskin kejadian ekstrem.** Bagian validasi (Des 2024 s/d Jul 2025)
  hanya memuat 4,7% jendela ekstrem dari 13 episode. Pemilihan epoch DLinear
  karenanya ditentukan terutama oleh perilaku di luar musim panas. Ini
  disengaja, karena itulah praktik standar yang sedang diuji.

### Batasan: status QAQC tidak seragam

| Periode | Status CDMO |
|---|---|
| 2021-2024 | final (`Historical=1`) |
| 2025 s/d 24 Jun 2026 08:45 | provisional plus, sudah QAQC sekunder (`ProvisionalPlus=1`) |
| 24 Jun 2026 09:00 s/d 27 Ags 2026 | provisional, belum QAQC sekunder (`ProvisionalPlus=0`) |

Di periode provisional tidak ada satu pun nilai DO yang ditolak atau ditandai
hilang, berbeda dengan tahun lain. Kriteria penyaringan juga berubah antar
tahun: flag `<1>` suspect hanya muncul di 2021 (9,6%). Bagian uji sama sekali
tidak memuat data final, dan 36 dari 53 episode D1 uji jatuh di periode
provisional. Episode itu bisa memuat pembacaan rendah akibat alat yang pada
data final akan ditolak. Dicatat sebagai batasan, eksperimen tidak dihentikan.
