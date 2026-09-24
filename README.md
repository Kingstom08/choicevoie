# Choos Voie

Choos Voie, video üzerinde dublaj yapmayı sağlayan bir dublaj oyunu / aracıdır. Bir video
yüklersin ya da YouTube'dan indirirsin, admin panelinde repliklerin (cue) başlangıç/bitiş
zamanlarını ve metnini tanımlarsın, ardından kullanıcı sekmesinde mikrofonla replik replik
kayıt yaparsın. Kayıtlar orijinal sesin üzerine bindirilerek montajlanır ve sonuç mp4 olarak
izlenip indirilebilir.

## Özellikler

- **Video ekleme** — dosya yükleme (.mp4 / .webm / .mov / .m4v) veya YouTube'dan indirme
  (en fazla 480p, h264+AAC tercih edilir; sadece youtube.com / youtu.be / m.youtube.com /
  music.youtube.com linkleri kabul edilir).
- **Admin panel** — replik ekleme/silme (sayfa yenilenmez, video başa sarmaz), başlangıç/bitiş
  saniyelerine tıklayınca video o ana gider, "Şu anki zamanı al" düğmeleri, replik süresi ve
  özet satırı.
- **Ses ayırma** — Demucs (kuruluysa) veya hafif merkez-kanal yöntemiyle konuşmayı arka
  plandan ayırır; yalnızca replik aralıkları işlenir. İlerleme yüzdesi, aşama metni, kalan
  süre ve cihaz rozeti gösterilir. Replikler sonradan değişirse "bayat" uyarısı verir.
- **Replikleri otomatik oluşturma** — YouTube altyazısından (yt-dlp ile sadece altyazı
  indirilir, zaman damgaları birebir) veya Whisper ile ses tanımadan; dil seçimi (otomatik
  algılama varsayılan) ve "ekle" / "hepsini değiştir" modu.
- **Transkribe Et** düğmesi — seçili zaman aralığını transkribe edip metin kutusunu doldurur.
- **Replikleri paylaşma** — JSON (uygulamaya geri yüklenebilir) veya SRT (standart altyazı)
  olarak indirme; dosya yükleyerek içe aktarma.
- **Dublaj kaydı** (`/dub/{id}`) — videonun zaman eksenli ses dalga şeridi (replik bölgeleri
  işaretli, oynatma çizgisi, tıklayınca o saniyeye atlar), mikrofonun anlık dalga formu ve
  seviye çubuğu, 3-2-1 geri sayım, hazırlık payı, otomatik sonraki replik, kayıt süresi
  uyarısı, mikrofon cihaz seçici, klavye kısayolları (`Boşluk` kaydet/durdur, `L` sahneyi
  dinle, `P` dublajlı dinle, `↑`/`↓` replik değiştir, `Esc` durdur). "Sahneyi Dinle" orijinal
  sesle çalar; kayıt sırasında video sesi varsayılan olarak kapalıdır. "Baştan Sona Önizle"
  ile montajdan önce tüm dublaj dinlenebilir.
- **Öncekiler ve sonuç** (`/takes/{id}`, `/take/{id}`) — kaydedilmiş dublajlar, durum
  rozetleri, montajlanmış mp4'ü izleme ve indirme.

## Gereksinimler

- Python 3.10+
- ffmpeg — ayrıca kurmana gerek yok; `imageio-ffmpeg` paketiyle gelir, uygulama ilk
  çalışmada kendi `bin/` klasörüne bağlar (sistemde ffmpeg varsa onu kullanır).
- Mikrofon
- Modern bir tarayıcı — **Chrome önerilir**. Safari'nin `MediaRecorder` desteği test
  edilmemiştir.

## Kurulum

Her platformda önce ortak adım çalıştırılır, sonra platforma özel ek adım eklenir.

### Ortak ilk adım (her platform)

```bash
pip install -r requirements.txt
```

### Windows / Linux + NVIDIA GPU

```bash
pip install -r requirements-cuda.txt
```

PyTorch'un CUDA sürümünü kurar. `faster-whisper`'ın GPU'da çalışması için
`nvidia-cublas-cu12` ve `nvidia-cudnn-cu12` de gerekir — bunlar aynı dosyada yer alır.
Uygulama bu DLL klasörlerini açılışta kendisi `PATH`'e ekler.

### macOS / Apple Silicon (M1–M4)

```bash
pip install -r requirements-mac.txt
```

PyTorch `demucs` ile birlikte gelir ve Metal (MPS) desteği hazırdır. `mlx-whisper`
Apple'ın Neural Engine'ini kullanarak transkripsiyonu hızlandırır; kuruluysa uygulama
otomatik olarak tercih eder, yoksa `faster-whisper` (CPU) kullanılır.

### Sadece CPU (herhangi bir platform)

Ek adım yok, `requirements.txt` yeterlidir.

## Çalıştırma

```bash
python -m uvicorn app:app --reload
```

Tarayıcıdan: http://127.0.0.1:8000

## Kullanım

1. **Video ekle** — ana sayfadan dosya yükle ya da YouTube linkini yapıştırıp indir.
2. **Repliği tanımla** — admin panelinde (`/admin/{id}`) videoyu izlerken başlangıç/bitiş
   saniyelerini gir ("Şu anki zamanı al" ile) ve metni yaz; istersen YouTube altyazısından
   veya Whisper ile otomatik oluştur, ya da JSON/SRT içe aktar.
3. **Ses ayır** (isteğe bağlı) — arka planı korumak istiyorsan "Ses Ayırma" kartından
   ayrıştırmayı başlat.
4. **Kaydet** — kullanıcı sekmesinde (`/dub/{id}`) her repliği mikrofonla kaydet; "Sahneyi
   Dinle" ile orijinali, "Baştan Sona Önizle" ile tüm dublajı kontrol et.
5. **Montajla** — kayıtlar tamamlandığında montaj başlatılır; "Arka planı koru" seçiliyse ve
   ayrıştırma yapılmışsa müzik/efekt replik aralığında korunur.
6. **İzle / indir** — "Öncekiler" (`/takes/{id}`) listesinden durumu takip et, sonuç mp4'ü
   `/take/{id}` üzerinden izle ve indir.

## Donanım hızlandırma

| İş | CUDA (NVIDIA) | Apple Silicon (MPS) | CPU |
|---|---|---|---|
| Demucs (ses ayırma) | Evet | Evet (Metal) | Evet |
| Whisper (otomatik replik / transkribe) | Evet (`faster-whisper`, float16) | Metal — `mlx-whisper` kuruluysa öncelikli | Evet (`faster-whisper`, int8) |
| ffmpeg montaj | CPU | CPU | CPU — `-c:v copy` ile video yeniden kodlanmaz |

CUDA denemesi VRAM yetersizliği gibi bir sebeple başarısız olursa otomatik olarak CPU'ya
düşülür.

## Ölçülmüş performans

Aşağıdaki ölçümler **RTX 3060 Laptop (6 GB)** üzerinde alınmıştır:

| İş | GPU (RTX 3060) | CPU |
|---|---|---|
| Demucs, 8 sn ses | 5 sn | 36.6 sn |
| Ayrıştırma, 70 sn video (43 sn replik) | 12.1 sn | — |
| Whisper small, 70 sn video | 2.7 sn | ~60 sn |

**Apple Silicon için ölçüm yapılmamıştır.** Aşağıdaki değerler tahmindir, gerçek
performans farklı olabilir:

- Demucs, MPS üzerinde CUDA'ya göre kabaca 2-4 kat daha yavaş olabilir.
- Whisper, `mlx-whisper` kullanıldığında CUDA süresine yakın olabilir.

## Veri ve gizlilik

`uploads/`, `outputs/` ve `choosvoie.db` depoya dahil değildir (`.gitignore`). Kendi
videolarını ve kayıtlarını paylaşma konusunda dikkatli ol; telif hakkı olan içeriği
yükleme/dağıtma sorumluluğu kullanıcıya aittir. YouTube indirme özelliği yalnızca hakkına
sahip olunan ya da indirmesine izin verilen içerikler için kullanılmalıdır.

## Bilinen sınırlar

- Kimlik doğrulama yoktur — uygulama yerel kullanım içindir.
- Montaj senkron çalışır; uzun videolarda istek montaj bitene kadar bekler.
- Mikrofon erişimi `localhost` dışında HTTPS gerektirir.
- Hafif (Demucs'suz) ses ayırma yöntemi stereo video gerektirir, mono videoda başarısız olur.
- YouTube otomatik altyazıları zaman damgası olarak kayan/tutarsız biçimde gelebilir.

## Proje yapısı

```
app.py                  # FastAPI uygulaması
requirements.txt        # ortak bağımlılıklar (tüm platformlar)
requirements-cuda.txt   # Windows/Linux + NVIDIA GPU ek bağımlılıkları
requirements-mac.txt    # macOS / Apple Silicon ek bağımlılıkları
templates/               # Jinja2 şablonları (index, admin, dub, takes, take, base, _tabs)
static/                  # stil ve istemci tarafı JS (harici kütüphane yok)
bin/                      # ffmpeg bağlantısı (ilk çalıştırmada oluşturulur)
uploads/                  # yüklenen videolar, uploads/rec (kayıtlar), uploads/bg (arka
                           # plan sesi), uploads/wave (dalga önbelleği)
outputs/                  # montajlanmış mp4 dosyaları
choosvoie.db            # SQLite veritabanı
```
