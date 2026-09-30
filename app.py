"""Choos Voie - basit video altyazı + dublaj oyunu uygulaması."""
import array
import glob
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from datetime import datetime
from urllib.parse import urlparse, quote

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import RedirectResponse, FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

try:
    # Windows sertifika deposunu Python'a tanit (yt-dlp SSL dogrulamasi icin)
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

try:
    from yt_dlp import YoutubeDL
    YTDLP_AVAILABLE = True
except Exception:
    YTDLP_AVAILABLE = False

def _add_nvidia_dll_path():
    """pip ile kurulan nvidia-* paketlerinin DLL klasorlerini PATH'e ekler.

    Windows'ta ctranslate2 (faster-whisper'in motoru) cublas/cudnn kutuphanelerini
    klasik PATH aramasiyla yukler. pip bu DLL'leri site-packages/nvidia/*/bin altina
    koyar ve orayi PATH'e eklemez; bu yuzden GPU "cublas64_12.dll bulunamadi" ile
    duser ve CPU'ya geri donerdi. Import'tan ONCE cagrilmali.
    """
    import glob
    import site
    dirs = []
    try:
        bases = list(site.getsitepackages())
        user_site = site.getusersitepackages()
        if isinstance(user_site, str):
            bases.append(user_site)
        else:
            bases.extend(user_site)
    except Exception:
        return
    for base in bases:
        dirs.extend(glob.glob(os.path.join(base, "nvidia", "*", "bin")))
    if not dirs:
        return
    os.environ["PATH"] = os.pathsep.join(dirs) + os.pathsep + os.environ.get("PATH", "")
    for d in dirs:
        try:
            os.add_dll_directory(d)
        except Exception:
            pass


_add_nvidia_dll_path()

try:
    # faster-whisper kurulu degilse (veya ctranslate2 bozuksa) uygulama cokmesin;
    # transkripsiyon/otomatik replik uclari anlamli Turkce hata donsun.
    from faster_whisper import WhisperModel
    import ctranslate2
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

# mlx-whisper sadece Apple Silicon'da anlamli (Metal/Neural Engine); varligini
# sadece tespit ediyoruz (import etmiyoruz - agir bir modul, gereksiz yere
# yuklenmesin). Gercek import kullanim aninda (mlx cagrisi icinde) yapilir.
MLX_AVAILABLE = importlib.util.find_spec("mlx_whisper") is not None


def _is_apple_silicon() -> bool:
    """mlx-whisper'in anlamli oldugu platform: macOS + Apple Silicon (arm64)."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_DIR = os.path.join(BASE_DIR, "bin")

# Proje bin/ klasorunu PATH'in BASINA ekle. Bazi bagimliliklar (ozellikle
# mlx_whisper/audio.py) sesi okumak icin dogrudan PATH'ten "ffmpeg" calistirir;
# biz ffmpeg'i bin/ icinde tuttugumuz icin PATH'te olmazsa o cagri
# FileNotFoundError ile patlar ve sessizce yavas yola dusulur.
# Platformdan bagimsizdir: Windows'ta ayni klasordeki ffmpeg.exe bulunur.
# (Ikilinin kendisi _ffmpeg_location() tarafindan tembel olusturulur; klasor
# o an yoksa PATH girdisi zararsizdir.)
os.environ["PATH"] = BIN_DIR + os.pathsep + os.environ.get("PATH", "")

UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
REC_DIR = os.path.join(UPLOADS_DIR, "rec")
WAVE_DIR = os.path.join(UPLOADS_DIR, "wave")
BG_DIR = os.path.join(UPLOADS_DIR, "bg")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
DB_PATH = os.path.join(BASE_DIR, "choosvoie.db")
ALLOWED_EXT = {".mp4", ".webm", ".mov", ".m4v"}
ALLOWED_HOSTS = {"youtube.com", "youtu.be", "m.youtube.com", "music.youtube.com"}
WAVE_BUCKETS = 1200

# Ses blob'u content-type -> uzantı eşlemesi (kayıt yüklerken kullanılır)
AUDIO_EXT_MAP = {
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/mp4": ".m4a",
    "audio/aac": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
}
MAX_REC_SIZE = 50 * 1024 * 1024  # 50 MB

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(REC_DIR, exist_ok=True)
os.makedirs(WAVE_DIR, exist_ok=True)
os.makedirs(BG_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Eski TTS sürümünden kalan klasörü temizle (varsa)
_old_tts_dir = os.path.join(BASE_DIR, "static", "tts")
if os.path.isdir(_old_tts_dir):
    shutil.rmtree(_old_tts_dir, ignore_errors=True)

app = FastAPI(title="Choos Voie")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Video basina ses ayirma islerinin bellek-ici durumu: {video_id: {"status", "method", "message"}}
SEPARATION_STATE = {}
SEPARATION_LOCK = threading.Lock()

# Video basina otomatik replik (whisper/youtube) islerinin bellek-ici durumu
AUTOCUE_STATE = {}
AUTOCUE_LOCK = threading.Lock()

# faster-whisper modeli tembel yuklenir, tek ornek olarak onbellekte tutulur.
WHISPER_MODEL = None
WHISPER_DEVICE = None
WHISPER_LOCK = threading.Lock()


def safe_filename(name: str) -> str:
    """Sadece dosya adını al (yol bileşenlerini at), path traversal engelle."""
    return os.path.basename(name or "")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()

    conn.execute(
        """CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            filename TEXT NOT NULL,
            created_at TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS cues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL,
            start_sec REAL NOT NULL,
            end_sec REAL NOT NULL,
            text TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS takes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL,
            name TEXT,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'draft',
            output_file TEXT,
            error TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            take_id INTEGER NOT NULL,
            cue_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(take_id, cue_id)
        )"""
    )

    # Ses ayirma (arka plan/konusma) icin ek kolonlar - sadece ADD COLUMN, veri kaybi yok.
    video_cols = [r["name"] for r in conn.execute("PRAGMA table_info(videos)").fetchall()]
    if "bg_file" not in video_cols:
        conn.execute("ALTER TABLE videos ADD COLUMN bg_file TEXT")
    if "bg_method" not in video_cols:
        conn.execute("ALTER TABLE videos ADD COLUMN bg_method TEXT")

    if "bg_cue_sig" not in video_cols:
        conn.execute("ALTER TABLE videos ADD COLUMN bg_cue_sig TEXT")

    if "bg_error" not in video_cols:
        conn.execute("ALTER TABLE videos ADD COLUMN bg_error TEXT")

    if "source_url" not in video_cols:
        conn.execute("ALTER TABLE videos ADD COLUMN source_url TEXT")

    take_cols = [r["name"] for r in conn.execute("PRAGMA table_info(takes)").fetchall()]
    if "keep_background" not in take_cols:
        conn.execute("ALTER TABLE takes ADD COLUMN keep_background INTEGER DEFAULT 0")

    # Arka plan ses seviyesi (yuzde, 0-150). Eski satirlarda varsayilan 100 = bugunku davranis.
    if "bg_volume" not in take_cols:
        conn.execute("ALTER TABLE takes ADD COLUMN bg_volume INTEGER DEFAULT 100")

    # Son basarili montaj zamani; eski satirlarda NULL (uyari gosterilmez).
    if "rendered_at" not in take_cols:
        conn.execute("ALTER TABLE takes ADD COLUMN rendered_at TEXT")

    # Son basarili montajda kullanilan kayit sayisi. rendered_at ile birlikte
    # "kayitlar degisti mi?" uyarisinda kayit SILME durumunu da yakalamak icin
    # (zaman damgasi karsilastirmasi tek basina azalan sayimi goremez).
    # Eski satirlarda NULL -> eski (zaman damgasi tabanli) davranisa duser.
    if "rendered_rec_count" not in take_cols:
        conn.execute("ALTER TABLE takes ADD COLUMN rendered_rec_count INTEGER")

    # Dublaj kayitlarinin master seviyesi (yuzde 0-200). Eski satirlar 100 = bugunku davranis.
    if "dub_volume" not in take_cols:
        conn.execute("ALTER TABLE takes ADD COLUMN dub_volume INTEGER DEFAULT 100")
    # Kayit seviyelerini otomatik esitle (0/1). Varsayilan kapali = davranis degismez.
    if "auto_level" not in take_cols:
        conn.execute("ALTER TABLE takes ADD COLUMN auto_level INTEGER DEFAULT 0")

    rec_cols = [r["name"] for r in conn.execute("PRAGMA table_info(recordings)").fetchall()]
    if "volume" not in rec_cols:
        conn.execute("ALTER TABLE recordings ADD COLUMN volume INTEGER DEFAULT 100")
    if "mean_db" not in rec_cols:
        conn.execute("ALTER TABLE recordings ADD COLUMN mean_db REAL")
    if "peak_db" not in rec_cols:
        conn.execute("ALTER TABLE recordings ADD COLUMN peak_db REAL")

    # Tek seferlik bakim: cue'su silinmis (yetim) kayitlar zaten kullanilamaz durumda
    # (render_take kayitlari JOIN cues ile okur), diskte ve DB'de yer kapliyorlar.
    orphans = conn.execute(
        "SELECT id, filename FROM recordings WHERE cue_id NOT IN (SELECT id FROM cues)"
    ).fetchall()
    if orphans:
        for r in orphans:
            p = os.path.join(REC_DIR, safe_filename(r["filename"]))
            if os.path.exists(p):
                os.remove(p)
        conn.execute("DELETE FROM recordings WHERE cue_id NOT IN (SELECT id FROM cues)")
        print(f"[bakim] {len(orphans)} yetim kayit temizlendi", file=sys.stderr)

    conn.commit()
    conn.close()


init_db()


# ---------------------------------------------------------------------------
# Sayfalar
# ---------------------------------------------------------------------------

@app.get("/")
def index(request: Request):
    conn = get_db()
    videos = conn.execute("SELECT * FROM videos ORDER BY id DESC").fetchall()
    conn.close()
    return templates.TemplateResponse(
        request, "index.html", {"videos": videos, "error": request.query_params.get("error")}
    )


@app.post("/upload")
async def upload_video(title: str = Form(...), video: UploadFile = File(...)):
    title = (title or "").strip()
    ext = os.path.splitext(video.filename or "")[1].lower()

    if not title:
        return RedirectResponse(url="/?error=Başlık boş olamaz", status_code=303)
    if ext not in ALLOWED_EXT:
        return RedirectResponse(url="/?error=Geçersiz dosya türü (mp4, webm, mov, m4v)", status_code=303)

    new_filename = f"{uuid.uuid4().hex}{ext}"
    dest_path = os.path.join(UPLOADS_DIR, new_filename)

    with open(dest_path, "wb") as f:
        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    conn = get_db()
    conn.execute(
        "INSERT INTO videos (title, filename, created_at) VALUES (?, ?, ?)",
        (title, new_filename, datetime.now().isoformat(timespec="seconds")),
    )
    video_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    conn.close()

    return RedirectResponse(url=f"/admin/{video_id}", status_code=303)


# ---------------------------------------------------------------------------
# YouTube'dan indirme (en fazla 480p)
# ---------------------------------------------------------------------------

def _ffmpeg_location():
    """yt-dlp'nin bulabilecegi bir ffmpeg klasoru dondur.

    Sistemde ffmpeg varsa onu kullanir. Yoksa imageio-ffmpeg ile gelen ikiliyi
    proje icindeki bin/ klasorune "ffmpeg.exe" adiyla baglar (yt-dlp dosya adina
    bakar, imageio'nun adi farklidir).
    """
    if shutil.which("ffmpeg"):
        return None  # PATH'te var, yt-dlp kendisi bulur

    try:
        import imageio_ffmpeg
        src = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None

    bin_dir = BIN_DIR
    os.makedirs(bin_dir, exist_ok=True)
    dst = os.path.join(bin_dir, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not os.path.exists(dst):
        try:
            os.link(src, dst)          # ayni diskte ise yer kaplamaz
        except Exception:
            try:
                shutil.copy2(src, dst)
            except Exception:
                return None
    return bin_dir


def _ffmpeg_bin():
    """Calistirilabilir ffmpeg ikilisinin tam yolunu (veya PATH ismini) dondur."""
    loc = _ffmpeg_location()
    if not loc:
        return "ffmpeg"
    return os.path.join(loc, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")


# yt-dlp/ffmpeg stderr'i TTY'ye bagliyken renk kacis kodu ("\x1b[0;31m") basar ve
# bu kodlar istisna metnine, oradan da URL'ye ve sayfaya sizar. Her zaman temizle.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _clean_error(msg) -> str:
    """Hata metnindeki ANSI renk kodlarini, "ERROR:" onekini ve fazla bosluklari at."""
    text = _ANSI_RE.sub("", str(msg))
    text = re.sub(r"^\s*ERROR:\s*", "", text)
    return " ".join(text.split())


# Sik gorulen yt-dlp hatalarini (kucuk harfe cevrilmis metinde aranir) kullaniciya
# anlasilir Turkce mesaja esler. Sirali: ilk eslesen kazanir.
_YT_ERROR_HINTS = (
    ("not a bot", "YouTube bot dogrulamasi istiyor. Bir sure sonra tekrar deneyin veya videoyu elle indirip yukleyin."),
    ("confirm your age", "Video yas sinirli; YouTube giris istiyor. Videoyu elle indirip yukleyin."),
    ("age-restricted", "Video yas sinirli; YouTube giris istiyor. Videoyu elle indirip yukleyin."),
    ("private video", "Video ozel (private); indirilemez."),
    ("members-only", "Video sadece kanal uyelerine acik; indirilemez."),
    ("removed by the uploader", "Video yukleyen tarafindan kaldirilmis."),
    ("account associated with this video has been terminated", "Videonun kanali kapatilmis."),
    ("available in your country", "Video bulundugunuz ulkede engelli (cografi kisit)."),
    ("blocked it in your country", "Video bulundugunuz ulkede engelli (cografi kisit)."),
    ("live event will begin", "Yayin henuz baslamamis; yayin bitince tekrar deneyin."),
    ("live event has not started", "Yayin henuz baslamamis; yayin bitince tekrar deneyin."),
    ("requested format is not available", "Bu video icin uygun bir 480p/mp4 format bulunamadi."),
    ("video is unavailable", "Video YouTube'da bulunamadi (kaldirilmis olabilir ya da baglanti hatali)."),
    ("video unavailable", "Video YouTube'da bulunamadi (kaldirilmis olabilir ya da baglanti hatali)."),
    ("this video is not available", "YouTube videoyu vermedi. Video kaldirilmis, ozel ya da bolgesel/telif kisitli olabilir."),
    ("unsupported url", "Bu baglanti desteklenmiyor."),
    ("certificate verify failed", "SSL sertifika dogrulamasi basarisiz (ag/proxy sorunu)."),
    ("urlopen error", "YouTube'a baglanilamadi; internet baglantinizi kontrol edin."),
    ("temporary failure in name resolution", "YouTube'a baglanilamadi; internet baglantinizi kontrol edin."),
    ("http error 429", "YouTube istekleri gecici olarak sinirladi (429). Biraz bekleyip tekrar deneyin."),
    ("sabr streaming", "YouTube bu videoyu indirilebilir formatta vermedi. Daha sonra tekrar deneyin."),
)


def _youtube_error_message(exc) -> str:
    """yt-dlp istisnasini kullaniciya gosterilecek Turkce mesaja cevir."""
    raw = _clean_error(exc)
    low = raw.lower()
    for needle, friendly in _YT_ERROR_HINTS:
        if needle in low:
            return friendly + " (Ayrinti: " + raw[:120] + ")"
    return "Video indirilemedi. (Ayrinti: " + raw[:150] + ")"


def _err(msg: str):
    return RedirectResponse(url="/?error=" + quote(_clean_error(msg)[:250]), status_code=303)


@app.post("/download")
def download_from_youtube(url: str = Form(...), title: str = Form("")):
    url = (url or "").strip()
    title = (title or "").strip()

    if not YTDLP_AVAILABLE:
        return _err("yt-dlp kurulu degil. Kurulum: pip install yt-dlp")

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if parsed.scheme not in ("http", "https") or host not in ALLOWED_HOSTS:
        return _err("Sadece YouTube linki kabul edilir")

    token = uuid.uuid4().hex
    opts = {
        # 480p'ye kadar en iyi video + en iyi ses; birlesemezse tek parca <=480p
        # once h264+aac (her tarayicida oynar), yoksa 480p'ye kadar ne varsa
        "format": ("bv*[height<=480][vcodec^=avc1]+ba[acodec^=mp4a]/"
                   "bv*[height<=480]+ba/b[height<=480]/bv*[height<=480]/b"),
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(UPLOADS_DIR, token + ".%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 2,
        # Hata metni kullaniciya gidiyor; yt-dlp ANSI renk kodu basmasin.
        "color": "no_color",
        # yt-dlp'nin varsayilan istemci seti (visionos/tv/web_embedded) bazi
        # videolarda yaniltici "This video is not available" veriyor; ayni video
        # android/web istemcisiyle sorunsuz cozuluyor. Varsayilani basta tutup
        # yedek istemci ekliyoruz (kaliteyi dusurmuyor, sadece yedek).
        "extractor_args": {"youtube": {"player_client": ["default", "android", "web"]}},
    }
    ffmpeg_dir = _ffmpeg_location()
    if ffmpeg_dir:
        opts["ffmpeg_location"] = ffmpeg_dir

    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as exc:
        return _err("Indirme basarisiz: " + _youtube_error_message(exc))

    downloaded = [f for f in os.listdir(UPLOADS_DIR) if f.startswith(token)]
    if not downloaded:
        return _err("Indirilen dosya bulunamadi")
    new_filename = downloaded[0]

    if not title:
        title = (info.get("title") or "YouTube videosu").strip()[:120]

    conn = get_db()
    conn.execute(
        "INSERT INTO videos (title, filename, created_at, source_url) VALUES (?, ?, ?, ?)",
        (title, new_filename, datetime.now().isoformat(timespec="seconds"), url),
    )
    video_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    conn.close()

    return RedirectResponse(url=f"/admin/{video_id}", status_code=303)


@app.get("/admin/{video_id}")
def admin_panel(request: Request, video_id: int):
    conn = get_db()
    v = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
    if not v:
        conn.close()
        return RedirectResponse(url="/?error=Video bulunamadı", status_code=303)
    cues = conn.execute(
        "SELECT * FROM cues WHERE video_id = ? ORDER BY start_sec ASC", (video_id,)
    ).fetchall()
    conn.close()
    return templates.TemplateResponse(
        request,
        "admin.html",
        {
            "video": v,
            "cues": cues,
            "error": request.query_params.get("error"),
            "ok": request.query_params.get("ok"),
        },
    )


# Cue sinirlarinda makul bir ust sinir: gercekci hicbir video 24 saati asmaz.
# inf/nan disinda, asiri buyuk sayisal degerleri de erkenden reddeder.
_MAX_CUE_SEC = 24 * 3600


def _clean_cue_fields(start_sec, end_sec, text):
    """(start, end, text, error) dondurur; error None ise gecerli.

    Kurallar ve mesajlar add_cue'nun eski gomulu dogrulamasiyla birebir aynidir.
    (import_cues kendi dogrulamasini kullanmaya devam eder: orada hatali satir
    atlanir, burada islem reddedilir.)
    """
    try:
        start = float(start_sec)
        end = float(end_sec)
    except (TypeError, ValueError):
        return None, None, (text or "").strip(), "Sayısal değer geçersiz"

    # inf/nan hicbir karsilastirmaya (start < 0, end <= start) takilmadan
    # gecerlilikten gecebilir (NaN karsilastirmalari hep False dondurur, inf
    # hicbir sinira carpmaz) ve DB'ye yazilip montajda cozulmemis OverflowError'a
    # ya da sessiz sonsuz susturmaya yol acar. Erken reddet.
    if not (math.isfinite(start) and math.isfinite(end)):
        return None, None, (text or "").strip(), "Sayısal değer geçersiz"
    if start > _MAX_CUE_SEC or end > _MAX_CUE_SEC:
        return None, None, (text or "").strip(), "Sayısal değer geçersiz"

    clean_text = (text or "").strip()
    if start < 0:
        return start, end, clean_text, "Başlangıç 0 veya üstü olmalı"
    if end <= start:
        return start, end, clean_text, "Bitiş, başlangıçtan büyük olmalı"
    if not clean_text:
        return start, end, clean_text, "Metin boş olamaz"
    return start, end, clean_text, None


@app.post("/admin/{video_id}/cue")
def add_cue(
    video_id: int,
    start_sec: float = Form(...),
    end_sec: float = Form(...),
    text: str = Form(...),
):
    start_sec, end_sec, text, err = _clean_cue_fields(start_sec, end_sec, text)
    if err:
        return RedirectResponse(url=f"/admin/{video_id}?error={err}", status_code=303)

    conn = get_db()
    v = conn.execute("SELECT id FROM videos WHERE id = ?", (video_id,)).fetchone()
    if not v:
        conn.close()
        return RedirectResponse(url="/?error=Video bulunamadı", status_code=303)

    conn.execute(
        "INSERT INTO cues (video_id, start_sec, end_sec, text) VALUES (?, ?, ?, ?)",
        (video_id, start_sec, end_sec, text),
    )
    conn.commit()
    conn.close()

    return RedirectResponse(url=f"/admin/{video_id}", status_code=303)


@app.post("/cue/{cue_id}/delete")
def delete_cue(cue_id: int):
    conn = get_db()
    try:
        cue = conn.execute("SELECT * FROM cues WHERE id = ?", (cue_id,)).fetchone()
        if not cue:
            return RedirectResponse(url="/?error=Cue bulunamadı", status_code=303)

        video_id = cue["video_id"]

        # Cue silinince ona bagli dublaj kayitlari yetim kalir (hicbir montaja giremez),
        # bu yuzden delete_video desenindeki gibi once dosyalari, sonra satirlari sil.
        recs = conn.execute("SELECT * FROM recordings WHERE cue_id = ?", (cue_id,)).fetchall()
        for r in recs:
            rec_path = os.path.join(REC_DIR, safe_filename(r["filename"]))
            if os.path.exists(rec_path):
                os.remove(rec_path)
        conn.execute("DELETE FROM recordings WHERE cue_id = ?", (cue_id,))

        conn.execute("DELETE FROM cues WHERE id = ?", (cue_id,))
        conn.commit()
    finally:
        conn.close()
    return RedirectResponse(url=f"/admin/{video_id}", status_code=303)


@app.post("/api/cue/{cue_id}/update")
def api_cue_update(
    cue_id: int,
    start_sec: str = Form(...),
    end_sec: str = Form(...),
    text: str = Form(...),
):
    """Var olan repligi guncelle. Cue id korunur, kayitlara DOKUNULMAZ."""
    new_start, new_end, clean_text, err = _clean_cue_fields(start_sec, end_sec, text)
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=400)

    conn = get_db()
    try:
        cue = conn.execute("SELECT * FROM cues WHERE id = ?", (cue_id,)).fetchone()
        if not cue:
            return JSONResponse({"ok": False, "error": "Cue bulunamadı"}, status_code=404)

        # Shift ucundaki ayni politika: bitis, video suresini asamaz (tumu-ya-hic,
        # kirpma yok). Sinir bilinmiyorsa (video dosyasi/onbellek yok) kontrol atlanir.
        video = conn.execute("SELECT * FROM videos WHERE id = ?", (cue["video_id"],)).fetchone()
        duration = _known_duration(video) if video else 0.0
        if duration > 0 and new_end > duration + 1e-6:
            return JSONResponse(
                {
                    "ok": False,
                    "error": f"Bitiş {new_end:.1f} sn, video süresini ({duration:.1f} sn) aşamaz.",
                },
                status_code=400,
            )

        old_start = float(cue["start_sec"])
        old_end = float(cue["end_sec"])
        timing_changed = abs(old_start - new_start) > 1e-6 or abs(old_end - new_end) > 1e-6

        affected = 0
        warning = None
        if timing_changed:
            affected = conn.execute(
                "SELECT COUNT(*) AS c FROM recordings WHERE cue_id = ?", (cue_id,)
            ).fetchone()["c"]
            if affected:
                old_dur = old_end - old_start
                new_dur = new_end - new_start
                warning = (
                    f"Bu repliğin {affected} dublaj kaydı var; "
                    f"süre {old_dur:.1f} sn → {new_dur:.1f} sn değişti. "
                    "Kayıtlar silinmedi ama artık aralığa tam uymayabilir."
                )

        conn.execute(
            "UPDATE cues SET start_sec = ?, end_sec = ?, text = ? WHERE id = ?",
            (new_start, new_end, clean_text, cue_id),
        )
        conn.commit()
    finally:
        conn.close()

    return JSONResponse(
        {
            "ok": True,
            "cue": {
                "id": cue_id,
                "video_id": cue["video_id"],
                "start_sec": new_start,
                "end_sec": new_end,
                "text": clean_text,
            },
            "affected_recordings": affected,
            "timing_changed": timing_changed,
            "warning": warning,
        }
    )


@app.post("/video/{video_id}/delete")
def delete_video(video_id: int):
    conn = get_db()
    v = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
    if not v:
        conn.close()
        return RedirectResponse(url="/?error=Video bulunamadı", status_code=303)

    takes = conn.execute("SELECT * FROM takes WHERE video_id = ?", (video_id,)).fetchall()
    take_ids = [t["id"] for t in takes]

    for t in takes:
        if t["output_file"]:
            out_path = os.path.join(OUTPUTS_DIR, safe_filename(t["output_file"]))
            if os.path.exists(out_path):
                os.remove(out_path)

    if take_ids:
        placeholders = ",".join("?" * len(take_ids))
        recs = conn.execute(
            f"SELECT * FROM recordings WHERE take_id IN ({placeholders})", take_ids
        ).fetchall()
        for r in recs:
            rec_path = os.path.join(REC_DIR, safe_filename(r["filename"]))
            if os.path.exists(rec_path):
                os.remove(rec_path)
        conn.execute(f"DELETE FROM recordings WHERE take_id IN ({placeholders})", take_ids)

    conn.execute("DELETE FROM takes WHERE video_id = ?", (video_id,))

    video_path = os.path.join(UPLOADS_DIR, safe_filename(v["filename"]))
    if os.path.exists(video_path):
        os.remove(video_path)

    wave_cache_path = os.path.join(WAVE_DIR, safe_filename(v["filename"]) + ".json")
    if os.path.exists(wave_cache_path):
        os.remove(wave_cache_path)

    if v["bg_file"]:
        bg_path = os.path.join(BG_DIR, safe_filename(v["bg_file"]))
        if os.path.exists(bg_path):
            os.remove(bg_path)

    conn.execute("DELETE FROM cues WHERE video_id = ?", (video_id,))
    conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))
    conn.commit()
    conn.close()

    with SEPARATION_LOCK:
        SEPARATION_STATE.pop(video_id, None)

    return RedirectResponse(url="/", status_code=303)


@app.get("/media/{filename}")
def media(filename: str):
    safe = safe_filename(filename)
    path = os.path.join(UPLOADS_DIR, safe)
    if not os.path.exists(path):
        return JSONResponse({"error": "Dosya bulunamadı"}, status_code=404)
    return FileResponse(path)


@app.get("/api/video/{video_id}/cues")
def api_cues(video_id: int):
    conn = get_db()
    cues = conn.execute(
        "SELECT * FROM cues WHERE video_id = ? ORDER BY start_sec ASC", (video_id,)
    ).fetchall()
    conn.close()
    result = [
        {"id": c["id"], "start_sec": c["start_sec"], "end_sec": c["end_sec"], "text": c["text"]}
        for c in cues
    ]
    return JSONResponse(result)


def _known_duration(video) -> float:
    """Video suresini ucuzdan pahaliya dogru bul; bulunamazsa 0.0."""
    filename = safe_filename(video["filename"])
    cache_path = os.path.join(WAVE_DIR, filename + ".json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                dur = float(json.load(f).get("duration") or 0.0)
            if dur > 0:
                return dur
        except Exception:
            pass  # onbellek bozuksa ffmpeg'e dus

    video_path = os.path.join(UPLOADS_DIR, filename)
    if os.path.exists(video_path):
        return _video_duration(_ffmpeg_bin(), video_path)
    return 0.0


@app.post("/api/video/{video_id}/cues/shift")
def api_cues_shift(video_id: int, offset_sec: str = Form(...), confirm: str = Form(None)):
    """Videodaki TUM repliklerin zamanini sabit bir offset kadar kaydir.

    Politika: tumu-ya-hicbiri. Sinira takilan bir kaydirma kirpilmaz, reddedilir;
    kirpma replikler arasi goreli senkronu bozar.
    """
    try:
        offset = float(offset_sec)
    except (TypeError, ValueError):
        return JSONResponse({"ok": False, "error": "Sayısal değer geçersiz"}, status_code=400)

    # inf/nan hicbir sinir kontrolune (asagidaki min/max karsilastirmalari) takilmayabilir
    # (ozellikle _known_duration() video dosyasi yokken 0 donunce ust sinir atlanir) ve
    # tum repliklerin start/end'i inf'e kayar. Erken reddet.
    if not math.isfinite(offset):
        return JSONResponse({"ok": False, "error": "Sayısal değer geçersiz"}, status_code=400)

    conn = get_db()
    try:
        v = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
        if not v:
            return JSONResponse({"ok": False, "error": "Video bulunamadı"}, status_code=404)

        cues = conn.execute(
            "SELECT * FROM cues WHERE video_id = ? ORDER BY start_sec ASC", (video_id,)
        ).fetchall()
        if not cues:
            return JSONResponse({"ok": False, "error": "Kaydırılacak replik yok"}, status_code=400)

        min_start = min(float(c["start_sec"]) for c in cues)
        max_end = max(float(c["end_sec"]) for c in cues)

        if min_start + offset < -1e-6:
            return JSONResponse(
                {
                    "ok": False,
                    "error": f"Kaydırma {-min_start:.1f} sn'den küçük olamaz: "
                             "ilk replik 0'ın altına düşer.",
                },
                status_code=400,
            )

        duration = _known_duration(v)
        if duration > 0 and max_end + offset > duration + 1e-6:
            max_offset = duration - max_end
            return JSONResponse(
                {
                    "ok": False,
                    "error": f"Kaydırma {max_offset:+.1f} sn'den büyük olamaz: "
                             f"son replik video süresini ({duration:.1f} sn) aşar.",
                },
                status_code=400,
            )

        rec_count = conn.execute(
            "SELECT COUNT(*) AS c FROM recordings WHERE cue_id IN "
            "(SELECT id FROM cues WHERE video_id = ?)",
            (video_id,),
        ).fetchone()["c"]
        if rec_count > 0 and (confirm or "").strip().lower() not in ("1", "on", "true"):
            return JSONResponse(
                {
                    "ok": False,
                    "needs_confirm": True,
                    "recording_count": rec_count,
                    "error": f"Bu videoda {rec_count} dublaj kaydı var; "
                             "tüm replikleri kaydırmak hepsinin senkronunu bozar.",
                },
                status_code=409,
            )

        # Tek ifade, atomik. Kayitlara dokunulmaz.
        cur = conn.execute(
            "UPDATE cues SET start_sec = start_sec + ?, end_sec = end_sec + ? WHERE video_id = ?",
            (offset, offset, video_id),
        )
        updated = cur.rowcount
        conn.commit()
        rows = conn.execute(
            "SELECT * FROM cues WHERE video_id = ? ORDER BY start_sec ASC", (video_id,)
        ).fetchall()
    finally:
        conn.close()

    return JSONResponse(
        {
            "ok": True,
            "offset_sec": offset,
            "updated": updated,
            "cues": [
                {"id": c["id"], "start_sec": c["start_sec"], "end_sec": c["end_sec"], "text": c["text"]}
                for c in rows
            ],
        }
    )


# ---------------------------------------------------------------------------
# Replik disa / ice aktarma (baska bir kullaniciyla ayni videoyu paylasmak icin)
# ---------------------------------------------------------------------------

MAX_IMPORT_SIZE = 1 * 1024 * 1024  # 1 MB
MAX_IMPORT_CUES = 500


def _export_filename(title: str, ext: str) -> str:
    safe_title = re.sub(r"[^\w\-]+", "_", title or "").strip("_") or "video"
    return f"{safe_title}-replikler.{ext}"


def _srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(float(seconds) * 1000)))
    h, rem = divmod(total_ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


@app.get("/video/{video_id}/cues/export.json")
def export_cues_json(video_id: int):
    conn = get_db()
    v = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
    if not v:
        conn.close()
        return JSONResponse({"error": "Video bulunamadı"}, status_code=404)
    cues = conn.execute(
        "SELECT * FROM cues WHERE video_id = ? ORDER BY start_sec ASC", (video_id,)
    ).fetchall()
    conn.close()

    payload = {
        "app": "choos-voie",
        "version": 1,
        "video_title": v["title"],
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "cue_count": len(cues),
        "cues": [
            {"start_sec": c["start_sec"], "end_sec": c["end_sec"], "text": c["text"]}
            for c in cues
        ],
    }
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    filename = _export_filename(v["title"], "json")
    return Response(
        content=body,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/video/{video_id}/cues/export.srt")
def export_cues_srt(video_id: int):
    conn = get_db()
    v = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
    if not v:
        conn.close()
        return JSONResponse({"error": "Video bulunamadı"}, status_code=404)
    cues = conn.execute(
        "SELECT * FROM cues WHERE video_id = ? ORDER BY start_sec ASC", (video_id,)
    ).fetchall()
    conn.close()

    lines = []
    for i, c in enumerate(cues, start=1):
        lines.append(str(i))
        lines.append(f"{_srt_timestamp(c['start_sec'])} --> {_srt_timestamp(c['end_sec'])}")
        lines.append(c["text"])
        lines.append("")
    body = "\n".join(lines)
    filename = _export_filename(v["title"], "srt")
    return Response(
        content=body,
        media_type="application/x-subrip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _parse_srt(text: str):
    """Basit SRT ayristirici. (start_sec, end_sec, text) uclulerinden liste dondurur."""
    blocks = re.split(r"\r?\n\r?\n+", text.strip())
    time_re = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")
    result = []
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip() != ""]
        if not lines:
            continue
        idx = 1 if re.match(r"^\d+$", lines[0].strip()) else 0
        if idx >= len(lines):
            continue
        m = time_re.search(lines[idx])
        if not m:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = m.groups()
        start = int(h1) * 3600 + int(m1) * 60 + int(s1) + int(ms1.ljust(3, "0")[:3]) / 1000.0
        end = int(h2) * 3600 + int(m2) * 60 + int(s2) + int(ms2.ljust(3, "0")[:3]) / 1000.0
        cue_text = " ".join(lines[idx + 1:]).strip()
        if cue_text:
            result.append((start, end, cue_text))
    return result


def _parse_import_file(raw: bytes):
    """Once JSON, olmazsa SRT olarak ayristirmayi dener. (cues, error) dondurur.

    cues: [(start_sec, end_sec, text), ...] — degerler henuz dogrulanmamis olabilir.
    """
    try:
        text = raw.decode("utf-8-sig")
    except Exception:
        return None, "Dosya UTF-8 olarak okunamadı"

    try:
        data = json.loads(text)
        cues_raw = data.get("cues") if isinstance(data, dict) else None
        if not isinstance(cues_raw, list):
            return None, "JSON içinde 'cues' listesi bulunamadı"
        cues = [(c.get("start_sec"), c.get("end_sec"), c.get("text")) for c in cues_raw if isinstance(c, dict)]
        return cues, None
    except Exception:
        pass

    srt_cues = _parse_srt(text)
    if not srt_cues:
        return None, "Dosya ne geçerli JSON ne de SRT olarak ayrıştırılabildi"
    return srt_cues, None


@app.post("/admin/{video_id}/cues/import")
async def import_cues(video_id: int, file: UploadFile = File(...), mode: str = Form("append")):
    mode = (mode or "append").strip().lower()
    if mode not in ("append", "replace"):
        return RedirectResponse(url=f"/admin/{video_id}?error=" + quote("Geçersiz mod"), status_code=303)

    conn = get_db()
    v = conn.execute("SELECT id FROM videos WHERE id = ?", (video_id,)).fetchone()
    if not v:
        conn.close()
        return RedirectResponse(url="/?error=" + quote("Video bulunamadı"), status_code=303)

    data = await file.read()
    if len(data) > MAX_IMPORT_SIZE:
        conn.close()
        return RedirectResponse(url=f"/admin/{video_id}?error=" + quote("Dosya 1MB sınırını aşıyor"), status_code=303)

    parsed, err = _parse_import_file(data)
    if err:
        conn.close()
        return RedirectResponse(url=f"/admin/{video_id}?error=" + quote(err), status_code=303)

    clean = []
    for start, end, text in parsed:
        try:
            start = float(start)
            end = float(end)
        except (TypeError, ValueError):
            continue
        # inf/nan disaridan gelen dosyada da ayni riski tasir (bkz. _clean_cue_fields);
        # burada semantik "atla" oldugu icin reddetmek yerine sadece satiri gec.
        if not (math.isfinite(start) and math.isfinite(end)):
            continue
        if start > _MAX_CUE_SEC or end > _MAX_CUE_SEC:
            continue
        text = (text or "").strip()
        if not text or start < 0 or end <= start:
            continue
        clean.append((start, end, text))

    if not clean:
        conn.close()
        return RedirectResponse(url=f"/admin/{video_id}?error=" + quote("Geçerli replik bulunamadı"), status_code=303)
    if len(clean) > MAX_IMPORT_CUES:
        conn.close()
        return RedirectResponse(
            url=f"/admin/{video_id}?error=" + quote("En fazla 500 replik içe aktarılabilir"), status_code=303
        )

    if mode == "replace":
        rec_count = conn.execute(
            """SELECT COUNT(*) AS c FROM recordings r
               JOIN takes t ON t.id = r.take_id WHERE t.video_id = ?""",
            (video_id,),
        ).fetchone()["c"]
        if rec_count > 0:
            conn.close()
            msg = (
                "Bu videoda kayıtlı dublaj sesleri var; replikleri değiştirmek onları "
                "sahipsiz bırakır. Önce dublajları sil ya da 'ekle' modunu kullan."
            )
            return RedirectResponse(url=f"/admin/{video_id}?error=" + quote(msg), status_code=303)
        conn.execute("DELETE FROM cues WHERE video_id = ?", (video_id,))

    for start, end, text in clean:
        conn.execute(
            "INSERT INTO cues (video_id, start_sec, end_sec, text) VALUES (?, ?, ?, ?)",
            (video_id, start, end, text),
        )
    conn.commit()
    conn.close()

    return RedirectResponse(
        url=f"/admin/{video_id}?ok=" + quote(f"{len(clean)} replik içe aktarıldı"), status_code=303
    )


# ---------------------------------------------------------------------------
# Otomatik konusma -> metin (faster-whisper) + tek aralik transkripsiyon
# ---------------------------------------------------------------------------

def _load_whisper_model(device: str):
    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel("small", device=device, compute_type=compute_type)


def _get_whisper(force_cpu: bool = False):
    """faster-whisper modelini tembel yukle, tek ornek olarak onbellekte tut.

    (model, kullanilan_cihaz) dondurur. force_cpu=True verilirse (CUDA calisirken
    hata alindiginda) model CPU ile yeniden yuklenir.
    """
    global WHISPER_MODEL, WHISPER_DEVICE
    with WHISPER_LOCK:
        if WHISPER_MODEL is not None and not (force_cpu and WHISPER_DEVICE == "cuda"):
            return WHISPER_MODEL, WHISPER_DEVICE

        if force_cpu:
            device = "cpu"
        else:
            try:
                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                device = "cpu"

        try:
            model = _load_whisper_model(device)
        except Exception:
            # CUDA/cuDNN yuklemesi patladiysa CPU ile bir kez daha dene.
            device = "cpu"
            model = _load_whisper_model(device)

        WHISPER_MODEL = model
        WHISPER_DEVICE = device
        return WHISPER_MODEL, WHISPER_DEVICE


def _whisper_transcribe(audio_path, language=None, vad_filter=False):
    """Segmentleri materyalize ederek (generator tembeldir) transkribe et.

    CUDA'da calisirken calisma-zamani hatasi alinirsa (VRAM/cuDNN) modeli CPU'ya
    dusurup bir kez daha dener. (segments_list, info, kullanilan_cihaz) dondurur.
    """
    model, device = _get_whisper()
    try:
        segments, info = model.transcribe(audio_path, language=language, vad_filter=vad_filter)
        return list(segments), info, device
    except Exception:
        if device != "cuda":
            raise
        model, device = _get_whisper(force_cpu=True)
        segments, info = model.transcribe(audio_path, language=language, vad_filter=vad_filter)
        return list(segments), info, device


MLX_MODEL_REPO = "mlx-community/whisper-small-mlx"


def _mlx_transcribe_raw(audio_path, language=None):
    """mlx_whisper.transcribe cagirir, ham sozlugu dondurur ({"text","language","segments"}).

    Import burada (cagri aninda) yapilir: MLX_AVAILABLE=False olan makinelerde
    (ornegin Windows) bu fonksiyon hic cagrilmadigi icin mlx_whisper hic
    yuklenmeye calisilmaz.
    """
    import mlx_whisper
    # mlx_whisper/audio.py sesi okumak icin PATH'ten "ffmpeg" calistirir. Proje
    # bin/ klasoru PATH'in basinda (bkz. BIN_DIR), ama icindeki ikili tembel
    # olusturuluyor; burada bir kez cagirarak dosyanin var oldugunu garanti et.
    _ffmpeg_bin()
    return mlx_whisper.transcribe(audio_path, path_or_hf_repo=MLX_MODEL_REPO, language=language)


def _engine_label(engine: str, device=None, mlx_error=None) -> str:
    """Arayuze gidecek motor etiketi.

    mlx yolundan faster-whisper'a dusulduyse bunu ETIKETTE gorunur kilar; aksi
    halde kullanici sadece "yavas calisiyor" diye fark ediyordu.
    """
    label = engine
    if device:
        label += f" ({str(device).upper()})"
    if mlx_error:
        label += f" — mlx başarısız: {_clean_error(mlx_error)[:120]}"
    return label


def _transcribe_file(wav_path, language=None):
    """Tek bir ses dosyasini metne cevirir. (text, detected_language, engine) dondurur.

    Apple Silicon'da mlx-whisper kuruluysa (Metal/Neural Engine - ctranslate2'nin
    desteklemedigi hizli yol) once onu dener; mlx cagrisi herhangi bir sebeple
    (ilk calistirmada model indirme hatasi, bellek, vb.) patlarsa hata yutulmadan
    (stderr'e loglanir) faster-whisper'a duser. Diger tum platformlarda / mlx
    kurulu degilse dogrudan mevcut faster-whisper yolu kullanilir (degismedi).
    /api/video/{id}/transcribe ucu bunu kullanir.
    """
    mlx_error = None
    if MLX_AVAILABLE and _is_apple_silicon():
        try:
            result = _mlx_transcribe_raw(wav_path, language=language)
            text = re.sub(r"\s+", " ", (result.get("text") or "").strip()).strip()
            detected_lang = result.get("language") or (language or "")
            return text, detected_lang, "mlx"
        except Exception as exc:
            mlx_error = exc
            print(f"[uyari] mlx-whisper basarisiz, faster-whisper'a dusuluyor: {exc}", file=sys.stderr)

    if not WHISPER_AVAILABLE:
        raise RuntimeError("faster-whisper kurulu değil. Kurulum: pip install faster-whisper")

    segments, info, device = _whisper_transcribe(wav_path, language=language, vad_filter=False)
    text = " ".join(s.text.strip() for s in segments if s.text and s.text.strip())
    text = re.sub(r"\s+", " ", text).strip()
    detected_lang = getattr(info, "language", None) or (language or "")
    return text, detected_lang, _engine_label("faster-whisper", device, mlx_error)


def _extract_wav_16k_mono(ffmpeg_bin, src_path, dst_path, start_sec=None, end_sec=None, timeout=300):
    """Kaynaktan (video ya da ayrik ses) 16kHz mono wav cikar (Whisper'in bekledigi format)."""
    cmd = [ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    if start_sec is not None:
        cmd += ["-ss", f"{start_sec:.3f}"]
    if end_sec is not None:
        cmd += ["-to", f"{end_sec:.3f}"]
    cmd += ["-i", src_path, "-vn", "-ac", "1", "-ar", "16000", dst_path]
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout)


def _vocal_source_path(video):
    """Ayrilmis bir vokal (konusma) kanali diskte varsa yolunu dondur, yoksa None.

    Mevcut ses ayirma ozelligi sadece arka plan (no_vocals) dosyasini kalici olarak
    saklar; vokal kanal su an saklanmiyor. Ileride ayni adlandirma kuraliyla
    saklanirsa burada otomatik kullanilir; yoksa cagiran video sesine duser.
    Yeni bir ayirma islemi BASLATMAZ, sadece diskte varsa kullanir.
    """
    candidate = os.path.join(BG_DIR, f"{safe_filename(video['filename'])}.vocals.m4a")
    if os.path.exists(candidate):
        return candidate
    return None


class TranscribeRequest(BaseModel):
    start_sec: float
    end_sec: float
    language: str = "auto"


@app.post("/api/video/{video_id}/transcribe")
def api_transcribe(video_id: int, payload: TranscribeRequest):
    if not WHISPER_AVAILABLE and not (MLX_AVAILABLE and _is_apple_silicon()):
        return JSONResponse({"ok": False, "error": "faster-whisper kurulu değil. Kurulum: pip install faster-whisper"})

    start_sec = payload.start_sec
    end_sec = payload.end_sec
    if start_sec < 0 or end_sec <= start_sec:
        return JSONResponse({"ok": False, "error": "Bitiş, başlangıçtan büyük olmalı"})
    dur = end_sec - start_sec
    if dur < 0.3 or dur > 120:
        return JSONResponse({"ok": False, "error": "Transkribe aralığı 0.3 - 120 saniye arasında olmalı"})

    conn = get_db()
    video = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
    conn.close()
    if not video:
        return JSONResponse({"ok": False, "error": "Video bulunamadı"})

    video_path = os.path.join(UPLOADS_DIR, safe_filename(video["filename"]))
    if not os.path.exists(video_path):
        return JSONResponse({"ok": False, "error": "Video dosyası bulunamadı"})

    src_path = _vocal_source_path(video) or video_path
    lang_in = (payload.language or "auto").strip().lower()
    language = None if lang_in == "auto" else lang_in

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_wav = os.path.join(tmp, "seg.wav")
        proc = _extract_wav_16k_mono(_ffmpeg_bin(), src_path, tmp_wav, start_sec, end_sec)
        if proc.returncode != 0 or not os.path.exists(tmp_wav) or os.path.getsize(tmp_wav) == 0:
            return JSONResponse({"ok": False, "error": "Ses aralığı çıkarılamadı: " + (proc.stderr or "")[-300:]})

        try:
            text, detected_lang, engine = _transcribe_file(tmp_wav, language=language)
        except Exception as exc:
            return JSONResponse({"ok": False, "error": "Transkripsiyon başarısız: " + str(exc)})

    return JSONResponse({
        "ok": True, "text": text, "language": detected_lang, "engine": engine,
        "took_sec": round(time.time() - t0, 2),
    })


# ---------------------------------------------------------------------------
# Otomatik replik uretimi (arka plan isi): YouTube altyazisi ya da Whisper
# ---------------------------------------------------------------------------

_VTT_TIME_RE = re.compile(
    r"(?:(\d+):)?(\d+):(\d+)\.(\d+)\s*-->\s*(?:(\d+):)?(\d+):(\d+)\.(\d+)"
)


def _vtt_ts_to_sec(h, m, s, ms) -> float:
    h = int(h) if h else 0
    return h * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000.0


def _clean_vtt_text(line: str) -> str:
    """VTT satirindaki etiketleri (<c>, inline zaman damgalari vb.) temizle."""
    return re.sub(r"<[^>]*>", "", line)


def _parse_vtt(text: str):
    """VTT altyazi metnini (start_sec, end_sec, text) uclulerine ayristirir.

    WEBVTT basligi, NOTE/STYLE/REGION bloklari ve inline etiketler atlanir.
    Otomatik altyazilarda ayni metin ardisik cue'larda tekrar edebilir; bu
    durumda ardisik ayni metinli cue'lar tek cue'da birlestirilir.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    n = len(lines)
    raw_cues = []
    i = 0
    while i < n:
        line = lines[i].strip()
        if line == "" or line.upper().startswith("WEBVTT") or line.startswith(("NOTE", "STYLE", "REGION")):
            i += 1
            continue
        m = _VTT_TIME_RE.search(line)
        if not m:
            i += 1
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = m.groups()
        start = _vtt_ts_to_sec(h1, m1, s1, ms1)
        end = _vtt_ts_to_sec(h2, m2, s2, ms2)
        i += 1
        text_lines = []
        while i < n and lines[i].strip() != "":
            text_lines.append(_clean_vtt_text(lines[i]))
            i += 1
        cue_text = re.sub(r"\s+", " ", " ".join(text_lines)).strip()
        if cue_text and (end - start) >= 0.3:
            raw_cues.append((start, end, cue_text))

    merged = []
    for start, end, txt in raw_cues:
        if merged and merged[-1][2] == txt:
            merged[-1] = (merged[-1][0], end, txt)
        else:
            merged.append((start, end, txt))
    return merged


def _update_autocue_state(video_id: int, **kwargs):
    with AUTOCUE_LOCK:
        st = dict(AUTOCUE_STATE.get(video_id, {}))
        st.update(kwargs)
        AUTOCUE_STATE[video_id] = st


def _autocue_from_youtube(video_id: int, language: str, url: str):
    """(cues, error) dondurur. cues: [(start_sec, end_sec, text), ...]. Video indirmez, sadece altyazi."""
    if not YTDLP_AVAILABLE:
        return None, "yt-dlp kurulu değil. Kurulum: pip install yt-dlp"

    conn = get_db()
    video = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
    conn.close()
    if not video:
        return None, "Video bulunamadı"

    url = (url or "").strip() or (video["source_url"] or "").strip()
    if not url:
        return None, "YouTube bağlantısı gerekli."

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if parsed.scheme not in ("http", "https") or host not in ALLOWED_HOSTS:
        return None, "Sadece YouTube linki kabul edilir"

    _update_autocue_state(video_id, stage="Altyazı indiriliyor", progress=5.0)

    lang = (language or "auto").strip().lower()
    sub_langs = []
    if lang and lang != "auto":
        sub_langs.append(lang)
    for base in ("en", "tr"):
        if base not in sub_langs:
            sub_langs.append(base)
    sub_langs.append(".*")  # yt-dlp subtitleslangs girdileri regex'tir; joker icin ".*" gerekir

    with tempfile.TemporaryDirectory() as tmp:
        token = uuid.uuid4().hex
        opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": sub_langs,
            "subtitlesformat": "vtt",
            "outtmpl": os.path.join(tmp, token + ".%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "retries": 2,
            "color": "no_color",
            "extractor_args": {"youtube": {"player_client": ["default", "android", "web"]}},
        }
        ffmpeg_dir = _ffmpeg_location()
        if ffmpeg_dir:
            opts["ffmpeg_location"] = ffmpeg_dir

        try:
            with YoutubeDL(opts) as ydl:
                ydl.extract_info(url, download=True)
            download_err = None
        except Exception as exc:
            # Birden fazla dil istendiginde (ornegin joker ".*") bir dil basarisiz
            # olsa bile (rate limit vb.) digerleri diske yazilmis olabilir; bu yuzden
            # hemen pes etmiyoruz, asagida diskte gercekten .vtt var mi kontrol ediyoruz.
            download_err = _youtube_error_message(exc)

        vtt_files = sorted(glob.glob(os.path.join(tmp, token + "*.vtt")))
        if not vtt_files:
            if download_err:
                return None, "Altyazı indirilemedi: " + download_err
            return None, "Bu videonun altyazısı yok. Whisper ile deneyebilirsin."

        chosen = vtt_files[0]
        if lang and lang != "auto":
            for f in vtt_files:
                if f".{lang}." in os.path.basename(f):
                    chosen = f
                    break

        _update_autocue_state(video_id, stage="Altyazı ayrıştırılıyor", progress=60.0)
        with open(chosen, "r", encoding="utf-8", errors="replace") as fh:
            vtt_text = fh.read()

    cues = _parse_vtt(vtt_text)
    _update_autocue_state(video_id, stage="Tamamlanıyor", progress=95.0)
    return cues, None


def _autocue_from_whisper(video_id: int, language: str):
    """(cues, error) dondurur. cues: [(start_sec, end_sec, text), ...].

    Motor secimi: Apple Silicon'da mlx-whisper kuruluysa (Metal/Neural Engine,
    cok daha hizli) once onu dener; mlx cagrisi basarisiz olursa (hata yutulmaz,
    stderr'e loglanir) faster-whisper'a duser. Diger platformlarda / mlx kurulu
    degilse dogrudan faster-whisper kullanilir (mevcut davranis, degismedi).
    mlx sonucu tek seferde (generator degil) geldigi icin o dalda ilerleme
    "Dinleniyor" asamasinda tek adimda ilerler; faster-whisper dalinda segment
    bazli gercek zamanli ilerleme aynen korunur.
    """
    if not WHISPER_AVAILABLE and not (MLX_AVAILABLE and _is_apple_silicon()):
        return None, "faster-whisper kurulu değil. Kurulum: pip install faster-whisper"

    conn = get_db()
    video = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
    conn.close()
    if not video:
        return None, "Video bulunamadı"

    video_path = os.path.join(UPLOADS_DIR, safe_filename(video["filename"]))
    if not os.path.exists(video_path):
        return None, "Video dosyası bulunamadı"

    src_path = _vocal_source_path(video) or video_path
    ffmpeg_bin = _ffmpeg_bin()
    duration = _video_duration(ffmpeg_bin, video_path)

    _update_autocue_state(video_id, stage="Ses çıkarılıyor", progress=2.0)

    lang = (language or "auto").strip().lower()
    whisper_lang = None if lang == "auto" else lang

    with tempfile.TemporaryDirectory() as tmp:
        tmp_wav = os.path.join(tmp, "full.wav")
        proc = _extract_wav_16k_mono(ffmpeg_bin, src_path, tmp_wav, timeout=1800)
        if proc.returncode != 0 or not os.path.exists(tmp_wav) or os.path.getsize(tmp_wav) == 0:
            return None, "Ses çıkarılamadı: " + (proc.stderr or "")[-300:]

        cues = None
        mlx_error = None
        if MLX_AVAILABLE and _is_apple_silicon():
            _update_autocue_state(video_id, stage="Dinleniyor (mlx)", progress=5.0, engine="mlx", device="mlx")
            try:
                result = _mlx_transcribe_raw(tmp_wav, language=whisper_lang)
                cues = []
                for seg in (result.get("segments") or []):
                    text = (seg.get("text") or "").strip()
                    if text:
                        cues.append((float(seg.get("start", 0.0)), float(seg.get("end", 0.0)), text))
                # mlx sonucu tek seferde gelir; ilerlemeyi tek adimda ilerlet.
                _update_autocue_state(video_id, stage="Dinleniyor 100%", progress=95.0)
            except Exception as exc:
                print(f"[uyari] mlx-whisper basarisiz, faster-whisper'a dusuluyor: {exc}", file=sys.stderr)
                mlx_error = exc
                cues = None

        if cues is None:
            if not WHISPER_AVAILABLE:
                return None, "faster-whisper kurulu değil. Kurulum: pip install faster-whisper"

            model, device = _get_whisper()
            _update_autocue_state(
                video_id, stage="Dinleniyor 0%", progress=5.0, device=device,
                engine=_engine_label("faster-whisper", device, mlx_error),
            )

            try:
                seg_gen, info = model.transcribe(tmp_wav, language=whisper_lang, vad_filter=True)
            except Exception:
                if device != "cuda":
                    raise
                model, device = _get_whisper(force_cpu=True)
                _update_autocue_state(
                    video_id, device=device,
                    engine=_engine_label("faster-whisper", device, mlx_error),
                )
                seg_gen, info = model.transcribe(tmp_wav, language=whisper_lang, vad_filter=True)

            total = duration if duration and duration > 0 else 1.0
            cues = []
            for seg in seg_gen:
                text = (seg.text or "").strip()
                if text:
                    cues.append((float(seg.start), float(seg.end), text))
                pct = min(100.0, (seg.end / total) * 100.0)
                _update_autocue_state(video_id, stage=f"Dinleniyor {int(pct)}%", progress=round(pct, 1))

    return cues, None


def _run_autocue_job(video_id: int, source: str, mode: str, language: str, url: str):
    if mode == "replace":
        conn = get_db()
        rec_count = conn.execute(
            """SELECT COUNT(*) AS c FROM recordings r
               JOIN takes t ON t.id = r.take_id WHERE t.video_id = ?""",
            (video_id,),
        ).fetchone()["c"]
        conn.close()
        if rec_count > 0:
            msg = (
                "Bu videoda kayıtlı dublaj sesleri var; replikleri değiştirmek onları "
                "sahipsiz bırakır. Önce dublajları sil ya da 'ekle' modunu kullan."
            )
            _update_autocue_state(video_id, status="error", message=msg)
            return

    try:
        if source == "youtube":
            cues, err = _autocue_from_youtube(video_id, language, url)
        else:
            cues, err = _autocue_from_whisper(video_id, language)
    except Exception as exc:
        cues, err = None, "Beklenmeyen hata: " + str(exc)

    if err:
        _update_autocue_state(video_id, status="error", message=err)
        return

    cues = cues or []
    note = ""
    if len(cues) > 500:
        cues = cues[:500]
        note = " (500 sınırı nedeniyle ilk 500 replik alındı)"

    # inf/nan disinda, asiri buyuk degerler de erkenden reddedilir (bkz. _MAX_CUE_SEC
    # yorumu ve import_cues'daki ayni kontrol). Whisper/YouTube kaynagi teorik olarak
    # boyle bir deger uretmemeli ama F1'in "tum yazma yollarini kapat" hedefi geregi
    # burada da dogrulaniyor. import_cues'daki "atla, reddetme" semantigi kullanilir:
    # gecersiz tek bir segment yuzunden toplu uretimin tamami cope atilmaz.
    clean_cues = []
    skipped = 0
    for start, end, text in cues:
        text = (text or "").strip()
        if not (math.isfinite(start) and math.isfinite(end)):
            skipped += 1
            continue
        if start > _MAX_CUE_SEC or end > _MAX_CUE_SEC:
            skipped += 1
            continue
        if not text or start < 0 or end <= start:
            skipped += 1
            continue
        clean_cues.append((start, end, text))

    if skipped:
        print(
            f"[uyari] oto-cue: {skipped} gecersiz segment atlandi (video_id={video_id})",
            file=sys.stderr,
        )

    conn = get_db()
    try:
        if mode == "replace":
            conn.execute("DELETE FROM cues WHERE video_id = ?", (video_id,))
        for start, end, text in clean_cues:
            conn.execute(
                "INSERT INTO cues (video_id, start_sec, end_sec, text) VALUES (?, ?, ?, ?)",
                (video_id, start, end, text),
            )
        conn.commit()
    finally:
        conn.close()

    _update_autocue_state(
        video_id, status="done", progress=100.0, stage="Tamamlandı",
        message="ok" + note, added=len(clean_cues),
    )


@app.post("/api/video/{video_id}/cues/auto")
def api_autocue_start(
    video_id: int,
    source: str = Form(...),
    mode: str = Form("append"),
    language: str = Form("auto"),
    url: str = Form(""),
):
    source = (source or "").strip().lower()
    mode = (mode or "append").strip().lower()
    if source not in ("youtube", "whisper"):
        return JSONResponse({"error": "Geçersiz kaynak"}, status_code=400)
    if mode not in ("append", "replace"):
        return JSONResponse({"error": "Geçersiz mod"}, status_code=400)

    conn = get_db()
    v = conn.execute("SELECT id FROM videos WHERE id = ?", (video_id,)).fetchone()
    conn.close()
    if not v:
        return JSONResponse({"error": "Video bulunamadı"}, status_code=404)

    with AUTOCUE_LOCK:
        current = AUTOCUE_STATE.get(video_id)
        if current and current.get("status") == "running":
            return JSONResponse({"status": "running"})
        AUTOCUE_STATE[video_id] = {
            "status": "running", "progress": 0.0, "stage": "Başlatılıyor",
            "message": "", "added": None, "source": source, "device": None, "engine": None,
        }

    threading.Thread(target=_run_autocue_job, args=(video_id, source, mode, language, url), daemon=True).start()
    return JSONResponse({"status": "running"})


@app.get("/api/video/{video_id}/cues/auto/status")
def api_autocue_status(video_id: int):
    with AUTOCUE_LOCK:
        st = dict(AUTOCUE_STATE.get(video_id, {}))
    return JSONResponse({
        "status": st.get("status", "none"),
        "progress": st.get("progress"),
        "stage": st.get("stage"),
        "message": st.get("message", ""),
        "added": st.get("added"),
        "source": st.get("source"),
        "device": st.get("device"),
        "engine": st.get("engine"),
    })


# ---------------------------------------------------------------------------
# Dublaj oyunu
# ---------------------------------------------------------------------------

@app.get("/dub/{video_id}")
def dub_page(request: Request, video_id: int):
    conn = get_db()
    v = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
    if not v:
        conn.close()
        return RedirectResponse(url="/?error=" + quote("Video bulunamadı"), status_code=303)

    cues = conn.execute(
        "SELECT * FROM cues WHERE video_id = ? ORDER BY start_sec ASC", (video_id,)
    ).fetchall()

    take = None
    take_param = request.query_params.get("take")
    if take_param:
        try:
            take = conn.execute(
                "SELECT * FROM takes WHERE id = ? AND video_id = ?", (int(take_param), video_id)
            ).fetchone()
        except ValueError:
            take = None
    conn.close()

    return templates.TemplateResponse(
        request,
        "dub.html",
        {"video": v, "cues": cues, "take": take, "error": request.query_params.get("error")},
    )


@app.post("/dub/{video_id}/take")
def create_take(video_id: int):
    conn = get_db()
    v = conn.execute("SELECT id FROM videos WHERE id = ?", (video_id,)).fetchone()
    if not v:
        conn.close()
        return JSONResponse({"error": "Video bulunamadı"}, status_code=404)

    conn.execute(
        "INSERT INTO takes (video_id, name, created_at, status, output_file, error) "
        "VALUES (?, NULL, ?, 'draft', NULL, NULL)",
        (video_id, datetime.now().isoformat(timespec="seconds")),
    )
    take_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    conn.close()
    return JSONResponse({"take_id": take_id})


def _audio_ext(content_type: str) -> str:
    key = (content_type or "").lower().split(";")[0].strip()
    return AUDIO_EXT_MAP.get(key, ".webm")


@app.post("/take/{take_id}/rec/{cue_id}")
async def upload_recording(take_id: int, cue_id: int, audio: UploadFile = File(...)):
    conn = get_db()
    take = conn.execute("SELECT * FROM takes WHERE id = ?", (take_id,)).fetchone()
    if not take:
        conn.close()
        return JSONResponse({"ok": False, "error": "Take bulunamadı"}, status_code=404)

    cue = conn.execute(
        "SELECT * FROM cues WHERE id = ? AND video_id = ?", (cue_id, take["video_id"])
    ).fetchone()
    if not cue:
        conn.close()
        return JSONResponse({"ok": False, "error": "Cue bulunamadı"}, status_code=404)

    data = bytearray()
    too_big = False
    while True:
        chunk = await audio.read(1024 * 1024)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > MAX_REC_SIZE:
            too_big = True
            break
    if too_big:
        conn.close()
        return JSONResponse({"ok": False, "error": "Dosya çok büyük (50MB sınırı)"}, status_code=400)
    if not data:
        conn.close()
        return JSONResponse({"ok": False, "error": "Boş ses kaydı"}, status_code=400)

    ext = _audio_ext(audio.content_type)
    new_filename = f"{uuid.uuid4().hex}{ext}"
    dest_path = os.path.join(REC_DIR, new_filename)
    with open(dest_path, "wb") as f:
        f.write(data)

    existing = conn.execute(
        "SELECT * FROM recordings WHERE take_id = ? AND cue_id = ?", (take_id, cue_id)
    ).fetchone()
    now = datetime.now().isoformat(timespec="seconds")
    if existing:
        old_path = os.path.join(REC_DIR, safe_filename(existing["filename"]))
        if os.path.exists(old_path):
            os.remove(old_path)
        # mean_db/peak_db sifirlanir: eski olcum yeni sesle artik gecersiz, bir
        # sonraki auto_level render'i yeniden olcecek. volume (kullanicinin bu
        # replik icin verdigi bilincli seviye tercihi) KORUNUR, sifirlanmaz.
        conn.execute(
            "UPDATE recordings SET filename = ?, created_at = ?, mean_db = NULL, peak_db = NULL WHERE id = ?",
            (new_filename, now, existing["id"]),
        )
    else:
        conn.execute(
            "INSERT INTO recordings (take_id, cue_id, filename, created_at) VALUES (?, ?, ?, ?)",
            (take_id, cue_id, new_filename, now),
        )
    conn.commit()
    conn.close()
    return JSONResponse({"ok": True, "audio_url": f"/rec/{new_filename}"})


@app.post("/take/{take_id}/rec/{cue_id}/delete")
def delete_recording(take_id: int, cue_id: int):
    conn = get_db()
    rec = conn.execute(
        "SELECT * FROM recordings WHERE take_id = ? AND cue_id = ?", (take_id, cue_id)
    ).fetchone()
    if rec:
        rec_path = os.path.join(REC_DIR, safe_filename(rec["filename"]))
        if os.path.exists(rec_path):
            os.remove(rec_path)
        conn.execute("DELETE FROM recordings WHERE id = ?", (rec["id"],))
        conn.commit()
    conn.close()
    return JSONResponse({"ok": True})


@app.get("/api/take/{take_id}")
def api_take(take_id: int):
    conn = get_db()
    take = conn.execute("SELECT * FROM takes WHERE id = ?", (take_id,)).fetchone()
    if not take:
        conn.close()
        return JSONResponse({"error": "Take bulunamadı"}, status_code=404)
    recs = conn.execute(
        "SELECT cue_id, filename FROM recordings WHERE take_id = ?", (take_id,)
    ).fetchall()
    conn.close()

    # DIKKAT: burada cue_id STR anahtar olarak kullanilir (dub.html:209-212 buna
    # bagimli, JSON.parse sonrasi tum obje anahtarlari string olur). take_detail()
    # ayni sozlugu INT anahtarla uretir (take.html:65 buna bagimli) - kasitli
    # fark, DEGISTIRME: ikisini de kirarsin.
    recordings = {str(r["cue_id"]): f"/rec/{r['filename']}" for r in recs}
    return JSONResponse(
        {
            "id": take["id"],
            "name": take["name"],
            "status": take["status"],
            "video_id": take["video_id"],
            "output_url": f"/output/{take['output_file']}" if take["output_file"] else None,
            "error": take["error"],
            "recordings": recordings,
            "keep_background": take["keep_background"],
            "bg_volume": take["bg_volume"],
            "dub_volume": take["dub_volume"],
            "auto_level": take["auto_level"],
        }
    )


def _set_take_status(take_id: int, status: str, output_file=None, error=None):
    conn = get_db()
    conn.execute(
        "UPDATE takes SET status = ?, output_file = ?, error = ? WHERE id = ?",
        (status, output_file, error, take_id),
    )
    conn.commit()
    conn.close()


def _fail_take(take_id: int, old_output, message: str):
    """Montaj hatasinda durumu 'error' yap ama onceki calisan ciktiyi KORU.

    Aksi halde basarisiz bir yeniden montaj, kullanicinin elindeki calisan
    videonun DB kaydini da silerdi (dosya diskte yetim kalirdi).
    """
    _set_take_status(take_id, "error", old_output, message)
    if old_output:
        print(f"[uyari] montaj basarisiz, onceki cikti korundu: {take_id}", file=sys.stderr)


def _video_has_audio(ffmpeg_bin: str, video_path: str) -> bool:
    """ffprobe yok; ffmpeg -i cikisinin stderr'inde Audio: stream'i ara.

    Bu komut cikti dosyasi verilmedigi icin ffmpeg exit code 1 ile biter,
    bu normaldir; sadece stderr metnine bakiyoruz.
    """
    try:
        proc = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-i", video_path],
            capture_output=True, text=True, errors="replace", timeout=30,
        )
    except Exception:
        return False
    for line in (proc.stderr or "").splitlines():
        s = line.strip()
        if s.startswith("Stream #0:") and "Audio:" in s:
            return True
    return False


def _parse_duration(stderr_text: str) -> float:
    """ffmpeg -i stderr'inden 'Duration: HH:MM:SS.ss' satirini ayikla."""
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr_text or "")
    if not m:
        return 0.0
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


# ---------------------------------------------------------------------------
# Kayit seviyesi olcumu + otomatik esitleme ("bazen az bazen fazla" varyansini
# gidermek icin). Master (dub_volume) ve kayit-basi manuel (recordings.volume)
# ile ayni carpimsal zincirde birlesir; bkz. render_take() gain enjeksiyonu.
# ---------------------------------------------------------------------------
AUTO_TARGET_DB = -20.0  # konusma icin tipik RMS hedefi
AUTO_MAX_BOOST_DB = 15.0
AUTO_MAX_CUT_DB = -6.0
AUTO_PEAK_CEIL_DB = -1.0
AUTO_SILENCE_FLOOR_DB = -45.0


def _measure_rec_level(ffmpeg_bin, path):
    """(mean_db, peak_db) dondur; olculemezse (None, None).

    silenceremove zorunlu: volumedetect ortalamayi tum dosya uzerinden alir,
    kayitta "hazirlik payi" sessizligi var, bu ortalamayi dusurur ve sessizligi
    cok olan klipler sistematik olarak fazla yukseltilir.
    """
    cmd = [ffmpeg_bin, "-hide_banner", "-nostdin", "-i", path,
           "-af", "silenceremove=start_periods=1:start_threshold=-50dB:"
                  "stop_periods=-1:stop_threshold=-50dB:stop_duration=0.2,volumedetect",
           "-f", "null", "-"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=60)
    except Exception:
        return None, None
    err = p.stderr or ""
    m = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?) dB", err)
    x = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?) dB", err)
    return (float(m.group(1)) if m else None), (float(x.group(1)) if x else None)


def _auto_gain_from(mean_db, peak_db) -> float:
    if mean_db is None or mean_db <= AUTO_SILENCE_FLOOR_DB:
        return 1.0
    g_db = max(AUTO_MAX_CUT_DB, min(AUTO_MAX_BOOST_DB, AUTO_TARGET_DB - mean_db))
    if peak_db is not None:
        g_db = min(g_db, AUTO_PEAK_CEIL_DB - peak_db)
    return 10.0 ** (g_db / 20.0)


def _auto_level_gains(ffmpeg_bin: str, recordings) -> list:
    """auto_level=1 iken her kayit icin otomatik kazanc (linear) hesapla.

    mean_db NULL olan kayitlar (hic olculmemis veya yeniden kayittan sonra
    sifirlanmis) olculur ve sonuc kendi kisa DB baglantisiyla yazilir
    (render_take'in conn'u bu noktada zaten kapali). mean_db zaten dolu olan
    kayitlar tekrar olculmez: auto_level kapaliyken maliyet 0, ilk auto
    render'da kayit basina ~30-60ms, sonraki render'larda 0ms.
    """
    gains = []
    conn = None
    for rec in recordings:
        mean_db = rec["rec_mean_db"]
        peak_db = rec["rec_peak_db"]
        if mean_db is None:
            path = os.path.join(REC_DIR, safe_filename(rec["rec_filename"]))
            mean_db, peak_db = _measure_rec_level(ffmpeg_bin, path)
            if conn is None:
                conn = get_db()
            conn.execute(
                "UPDATE recordings SET mean_db = ?, peak_db = ? WHERE id = ?",
                (mean_db, peak_db, rec["rec_id"]),
            )
            if mean_db is None:
                print(f"[seviye] olcum basarisiz rec={rec['rec_id']} (gain=1.0)", file=sys.stderr)
            else:
                g = _auto_gain_from(mean_db, peak_db)
                g_db = 20.0 * math.log10(g) if g > 0 else 0.0
                pk_display = peak_db if peak_db is not None else float("nan")
                print(
                    f"[seviye] rec={rec['rec_id']} mean={mean_db:.1f}dB "
                    f"peak={pk_display:.1f}dB gain={g_db:+.1f}dB",
                    file=sys.stderr,
                )
        gains.append(_auto_gain_from(mean_db, peak_db))
    if conn is not None:
        conn.commit()
        conn.close()
    return gains


def _effective_dub_gain(dub_master: float, rec, auto_gains, idx: int) -> float:
    """Uc katmanli carpimsal kazanc: master x kayit-basi manuel x otomatik."""
    rec_vol = max(0, min(200, int(rec["rec_volume"] if rec["rec_volume"] is not None else 100))) / 100.0
    auto_g = auto_gains[idx] if auto_gains is not None else 1.0
    return dub_master * rec_vol * auto_g


def _compute_waveform(ffmpeg_bin: str, video_path: str) -> dict:
    """Video icin sabit uzunlukta (WAVE_BUCKETS) tepe (peak) dalga formu uret.

    ffprobe yok; sure ffmpeg -i stderr'inden, PCM ise stdout'tan okunur.
    Hata / ses kanali yoksa peaks hepsi 0.0 olacak sekilde 200 ile sonuc doner.
    """
    def empty(dur=0.0, no_audio=True):
        return {"duration": round(dur, 3), "peaks": [0.0] * WAVE_BUCKETS, "count": WAVE_BUCKETS, "no_audio": no_audio}

    duration = 0.0
    try:
        probe = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-i", video_path],
            capture_output=True, text=True, errors="replace", timeout=30,
        )
        duration = _parse_duration(probe.stderr)
    except Exception:
        duration = 0.0

    if not _video_has_audio(ffmpeg_bin, video_path):
        return empty(duration, True)

    # Cok uzun videolarda bellegi kontrol altinda tutmak icin ornek hizini dusur.
    sample_rate = 4000 if duration > 3 * 3600 else 8000

    try:
        proc = subprocess.run(
            [ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "error",
             "-i", video_path, "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-"],
            capture_output=True, timeout=600,
        )
    except Exception:
        return empty(duration, True)

    if proc.returncode != 0 or not proc.stdout:
        return empty(duration, True)

    data = proc.stdout
    if len(data) % 2 == 1:
        data = data[:-1]
    samples = array.array("h", data)
    total = len(samples)
    if total == 0:
        return empty(duration, True)

    if duration <= 0:
        duration = total / float(sample_rate)

    bucket_size = max(1, total // WAVE_BUCKETS)
    peaks = []
    for i in range(WAVE_BUCKETS):
        start = i * bucket_size
        end = total if i == WAVE_BUCKETS - 1 else min(start + bucket_size, total)
        if start >= total:
            peaks.append(0.0)
            continue
        chunk = samples[start:end]
        peak = max((abs(x) for x in chunk), default=0)
        peaks.append(round(min(peak / 32768.0, 1.0), 4))

    return {"duration": round(duration, 3), "peaks": peaks, "count": WAVE_BUCKETS, "no_audio": False}


@app.get("/api/video/{video_id}/waveform")
def api_waveform(video_id: int):
    conn = get_db()
    v = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
    conn.close()
    if not v:
        return JSONResponse({"error": "Video bulunamadı"}, status_code=404)

    filename = safe_filename(v["filename"])
    cache_path = os.path.join(WAVE_DIR, filename + ".json")

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return JSONResponse(json.load(f))
        except Exception:
            pass  # önbellek bozuksa yeniden hesapla

    video_path = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(video_path):
        return JSONResponse({"duration": 0.0, "peaks": [0.0] * WAVE_BUCKETS, "count": WAVE_BUCKETS, "no_audio": True})

    result = _compute_waveform(_ffmpeg_bin(), video_path)

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f)
    except Exception:
        pass  # önbellek yazılamazsa sonucu yine de döndür

    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Ses ayirma (konusma / arka plan) - "ikisi birden": Demucs kuruluysa o, degilse
# hafif merkez-kanal-iptali yontemi kullanilir.
# ---------------------------------------------------------------------------

def _demucs_available() -> bool:
    return importlib.util.find_spec("demucs") is not None


_ALIMITER_CACHE = None


def _has_alimiter(ffmpeg_bin: str) -> bool:
    """Bu ffmpeg build'inde alimiter filtresi var mi? Modul seviyesinde onbelleklenir.

    Zorunlu probe: alimiter'i olmayan bir build'de her iki render denemesi de
    (-c:v copy ve libx264 fallback) filter hatasiyla duserdi.
    """
    global _ALIMITER_CACHE
    if _ALIMITER_CACHE is None:
        try:
            p = subprocess.run(
                [ffmpeg_bin, "-hide_banner", "-filters"],
                capture_output=True, text=True, errors="replace", timeout=20,
            )
            _ALIMITER_CACHE = " alimiter " in (p.stdout or "")
        except Exception:
            _ALIMITER_CACHE = False
    return _ALIMITER_CACHE


def _torch_device() -> str:
    """En iyi kullanilabilir torch cihazini sirayla dondur: 'cuda' -> 'mps' -> 'cpu'.

    torch kurulumu arka planda devam ediyor olabilir; import basarisiz olursa
    (henuz kurulmamis / bozuk) hic hata firlatmadan 'cpu' varsayilir. MPS
    (Apple Silicon GPU) icin hem is_available() hem is_built() kontrol edilir;
    bu oznitelikler Windows/Linux derlemelerinde de var ama False donebilir,
    yine de ayri try/except icinde denenir ki torch.backends.mps beklenmedik
    bir sekilde eksik/farkli olsa bile _torch_device() cokmeden 'cpu'ya dussun.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _update_separation_state(video_id: int, **kwargs):
    with SEPARATION_LOCK:
        st = dict(SEPARATION_STATE.get(video_id, {}))
        st.update(kwargs)
        SEPARATION_STATE[video_id] = st


def _video_duration(ffmpeg_bin: str, video_path: str) -> float:
    try:
        proc = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-i", video_path],
            capture_output=True, text=True, errors="replace", timeout=30,
        )
    except Exception:
        return 0.0
    return _parse_duration(proc.stderr)


def _merge_cue_segments(cues, duration: float, pad: float = 0.5, gap: float = 1.0):
    """Cue araliklarina pad ekle, cakisan/bitisik (aralari gap'ten kucuk) araliklari birlestir.

    (start_sec, end_sec) listesi dondurur, baslangica gore sirali.
    """
    ranges = []
    for c in cues:
        s = max(0.0, float(c["start_sec"]) - pad)
        e = float(c["end_sec"]) + pad
        if duration > 0:
            e = min(e, duration)
        if e > s:
            ranges.append((s, e))
    ranges.sort()

    merged = []
    for s, e in ranges:
        if merged and s - merged[-1][1] < gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _cue_signature(cues) -> str:
    """Cue baslangic/bitis ciftlerinden (metin haric) sha256 imza uret."""
    pairs = sorted((round(float(c["start_sec"]), 3), round(float(c["end_sec"]), 3)) for c in cues)
    raw = json.dumps(pairs, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _run_demucs_batch(video_id, seg_paths, seg_durations, tmp_root, device, total_dur, stage_prefix):
    """Tum segment wav dosyalarini TEK demucs cagrisinda isler.

    Ayri ayri her segment icin `python -m demucs` calistirmak, model yuklemesini
    (birkaç saniye) segment sayisi kadar tekrarlar ve kazanci yer yer yutar; bunun
    yerine tum segmentler tek subprocess cagrisina verilir (model bir kez yuklenir).
    Gercek ilerleme icin ayri bir izleyici thread, demucs'un her segment icin
    diske yazdigi `no_vocals.wav` cikti dosyasinin VARLIGINI kontrol eder
    (stderr parse edilmez — kirilgan olmaz).

    GPU denemesi (CUDA ya da Apple Silicon MPS) basarisiz olursa (VRAM/bellek
    yetmedi, MPS'in henuz desteklemedigi bir islem vb.) TUM batch CPU ile bir
    kez daha denenir.
    (ok, kullanilan_cihaz, fallback_oldu_mu, hata_metni_veya_None) dondurur; basariliysa
    her segment icin beklenen cikti yolu `expected` sirasiyla diskte mevcuttur.
    Hata durumunda hata metninin basina hangi cihazin denendigi eklenir.
    """
    expected = []
    for p in seg_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        expected.append(os.path.join(tmp_root, "htdemucs", stem, "no_vocals.wav"))

    n_total = len(seg_paths)
    attempts = [device, "cpu"] if device in ("cuda", "mps") else [device]
    last_err = None

    for attempt_device in attempts:
        cmd = [sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs", "-d", attempt_device]
        if attempt_device in ("cuda", "mps"):
            cmd += ["--segment", "7"]  # GPU bellek tasmasini onler (6 GB VRAM / Apple GPU icin)
        cmd += ["-o", tmp_root, *seg_paths]

        start_time = time.time()
        stop_event = threading.Event()

        def _monitor():
            seen = set()
            while not stop_event.is_set():
                for i, out_path in enumerate(expected):
                    if i not in seen and os.path.exists(out_path):
                        seen.add(i)
                        done_dur = sum(seg_durations[j] for j in seen)
                        elapsed = time.time() - start_time
                        remaining = max(0.0, total_dur - done_dur)
                        rate = elapsed / done_dur if done_dur > 0 else None
                        eta = int(remaining * rate) if rate else None
                        _update_separation_state(
                            video_id,
                            stage=f"{stage_prefix} {len(seen)}/{n_total}",
                            progress=round(5 + done_dur / total_dur * 90, 1),
                            eta_sec=eta,
                        )
                time.sleep(0.4)

        mon = threading.Thread(target=_monitor, daemon=True)
        mon.start()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=3600)
        finally:
            stop_event.set()
            mon.join(timeout=2)

        if proc.returncode == 0 and all(os.path.exists(p) for p in expected):
            return True, attempt_device, attempt_device != device, None
        last_err = f"[{attempt_device}] " + (proc.stderr or "")

    return False, attempts[-1], attempts[-1] != device, last_err


def _audio_is_stereo(ffmpeg_bin: str, video_path: str) -> bool:
    """ffmpeg -i stderr'inde 'Audio:' satirinda 'stereo' geciyor mu bak."""
    try:
        proc = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-i", video_path],
            capture_output=True, text=True, errors="replace", timeout=30,
        )
    except Exception:
        return False
    for line in (proc.stderr or "").splitlines():
        s = line.strip()
        if s.startswith("Stream #0:") and "Audio:" in s:
            return "stereo" in s
    return False


def _set_bg_error(video_id: int, message: str):
    """Son ayristirma hatasini DB'ye yaz (sunucu yeniden baslasa da kaybolmasin)."""
    try:
        conn = get_db()
        conn.execute("UPDATE videos SET bg_error = ? WHERE id = ?", (message, video_id))
        conn.commit()
        conn.close()
    except Exception:
        pass


def separate_audio(video_id: int):
    """Video icin arka plan/konusma ayrimi yap. (ok, method, message) dondurur.

    Sadece replik araliklarinda (0.5 sn paylı, birlesik) ayristirma yapar —
    tam video degil. Demucs kuruluysa onu kullanir (yuksek kalite ayrim,
    CUDA varsa GPU'da). Kurulu degilse, stereo videolarda merkez kanal iptali
    ile hafif bir ayrim yapar (mono videolarda bu yontem calismaz).
    Hicbir zaman exception firlatmaz.

    YIKICI DEGIL: yeni birlesik cikti once gecici bir dosyaya yazilir; sadece
    TUM adimlar (cikarma, ayristirma, birlestirme, sure kontrolu) basarili
    olursa mevcut calisan bg dosyasinin yerine ATOMIK olarak (os.replace)
    konur. Herhangi bir adim basarisiz olursa eski dosya ve DB alanlari
    (bg_file/bg_method/bg_cue_sig) OLDUGU GIBI kalir; hata bg_error kolonuna
    ve SEPARATION_STATE'e yazilir.
    """
    conn = get_db()
    video = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
    cues = conn.execute(
        "SELECT * FROM cues WHERE video_id = ? ORDER BY start_sec ASC", (video_id,)
    ).fetchall()
    conn.close()
    if not video:
        return False, "", "Video bulunamadı"
    if not cues:
        msg = "Önce replik eklemelisin — ayrıştırma sadece replik aralıklarında yapılır."
        _set_bg_error(video_id, msg)
        return False, "", msg

    video_path = os.path.join(UPLOADS_DIR, safe_filename(video["filename"]))
    if not os.path.exists(video_path):
        msg = "Video dosyası bulunamadı"
        _set_bg_error(video_id, msg)
        return False, "", msg

    ffmpeg_bin = _ffmpeg_bin()
    method = "demucs" if _demucs_available() else "light"
    device = _torch_device() if method == "demucs" else "cpu"
    _update_separation_state(video_id, device=device, stage="Süre hesaplanıyor", progress=0.0, eta_sec=None)

    def _fail(msg):
        _set_bg_error(video_id, msg)
        return False, method, msg

    duration = _video_duration(ffmpeg_bin, video_path)
    segments = _merge_cue_segments(cues, duration)
    if duration <= 0 and segments:
        duration = max(e for _, e in segments)
    if not segments:
        return _fail("Replik aralıkları geçersiz")

    bg_filename = f"{safe_filename(video['filename'])}.bg.m4a"
    bg_path = os.path.join(BG_DIR, bg_filename)
    bg_new_filename = f"{safe_filename(video['filename'])}.bg.new.m4a"
    bg_new_path = os.path.join(BG_DIR, bg_new_filename)

    seg_durations = [e - s for s, e in segments]
    total_dur = sum(seg_durations) or 1.0
    n = len(segments)
    fallback_note = ""

    if method == "light" and not _audio_is_stereo(ffmpeg_bin, video_path):
        return _fail(
            "Videonun sesi mono — hafif yöntem stereo gerektirir. "
            "Demucs kurarak deneyebilirsin: pip install demucs"
        )

    try:
        with tempfile.TemporaryDirectory() as tmp:
            # 1) Once TUM segmentleri video'dan cikart (hizli, tekil ffmpeg cagrilari).
            #    Extraction agirligi toplam ilerlemenin 0-5%'i.
            seg_ins = []
            for i, (s, e) in enumerate(segments, start=1):
                _update_separation_state(video_id, stage=f"Ses çıkarılıyor {i}/{n}",
                                          progress=round((i - 1) / n * 5, 1))
                seg_in = os.path.join(tmp, f"seg_{i}.wav")
                proc = subprocess.run(
                    [ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                     "-ss", f"{s:.3f}", "-to", f"{e:.3f}", "-i", video_path,
                     "-vn", "-ac", "2", "-ar", "44100", seg_in],
                    capture_output=True, text=True, errors="replace", timeout=300,
                )
                if proc.returncode != 0 or not os.path.exists(seg_in) or os.path.getsize(seg_in) == 0:
                    return _fail("Ses çıkarılamadı: " + (proc.stderr or "")[-400:])
                seg_ins.append(seg_in)

            # 2) Ayristirma: demucs icin TEK bir subprocess cagrisinda tum segmentler
            #    islenir (model bir kez yuklenir — segment basina tekrar tekrar model
            #    yuklemek kazanci yutar). Hafif yontemde segment basina ffmpeg zaten ucuz.
            out_wavs = []  # (start_sec, path)
            if method == "demucs":
                ok_batch, used_device, fell_back, err = _run_demucs_batch(
                    video_id, seg_ins, seg_durations, tmp, device, total_dur, "Ayrıştırılıyor"
                )
                if not ok_batch:
                    # _run_demucs_batch hata metninin basina denenen cihazi ekler (ör. "[mps] ...").
                    return _fail("Demucs çalıştırılamadı: " + (err or "")[-400:])
                if fell_back:
                    failed_device = device
                    device = "cpu"
                    fallback_note = f" ({failed_device.upper()} başarısız, CPU'ya geçildi)"
                    _update_separation_state(video_id, device=device)
                for (s, _e), seg_in in zip(segments, seg_ins):
                    stem = os.path.splitext(os.path.basename(seg_in))[0]
                    out_wavs.append((s, os.path.join(tmp, "htdemucs", stem, "no_vocals.wav")))
            else:
                done_dur = 0.0
                start_time = time.time()
                for i, ((s, _e), seg_in) in enumerate(zip(segments, seg_ins), start=1):
                    seg_out = os.path.join(tmp, f"seg_{i}_bg.wav")
                    lproc = subprocess.run(
                        [ffmpeg_bin, "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
                         "-i", seg_in, "-af", "pan=stereo|c0=c0-c1|c1=c1-c0", seg_out],
                        capture_output=True, text=True, errors="replace", timeout=120,
                    )
                    if lproc.returncode != 0 or not os.path.exists(seg_out) or os.path.getsize(seg_out) == 0:
                        return _fail("Ayrıştırma başarısız: " + (lproc.stderr or "")[-400:])
                    done_dur += seg_durations[i - 1]
                    elapsed_total = time.time() - start_time
                    remaining_dur = max(0.0, total_dur - done_dur)
                    rate = elapsed_total / done_dur if done_dur > 0 else None
                    eta = int(remaining_dur * rate) if rate else None
                    _update_separation_state(
                        video_id, stage=f"Ayrıştırılıyor {i}/{n}",
                        progress=round(5 + done_dur / total_dur * 90, 1), eta_sec=eta,
                    )
                    out_wavs.append((s, seg_out))

            for _s, p in out_wavs:
                if not os.path.exists(p) or os.path.getsize(p) == 0:
                    return _fail("Ayrıştırma çıktısı bulunamadı veya boş: " + p)

            _update_separation_state(video_id, stage="Birleştiriliyor", progress=95.0, eta_sec=0)

            input_args = []
            filter_parts = []
            mix_labels = []
            for idx, (s, path) in enumerate(out_wavs):
                input_args += ["-i", path]
                delay_ms = int(round(s * 1000))
                filter_parts.append(f"[{idx}:a]adelay={delay_ms}|{delay_ms}[a{idx}]")
                mix_labels.append(f"[a{idx}]")
            mix_expr = (
                "".join(mix_labels)
                + f"amix=inputs={len(out_wavs)}:duration=longest:normalize=0,apad,atrim=0:{duration:.3f}[aout]"
            )
            filter_complex = ";".join(filter_parts + [mix_expr])

            # ONEMLI: cikti dogrudan calisan bg_path'in UZERINE degil, gecici bir
            # dosyaya (bg_new_path) yazilir. Eski dosya sadece asagida TUM kontroller
            # basarili olduktan sonra atomik os.replace ile degistirilir.
            if os.path.exists(bg_new_path):
                try:
                    os.remove(bg_new_path)
                except OSError:
                    pass
            merge_proc = subprocess.run(
                [ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                 *input_args, "-filter_complex", filter_complex, "-map", "[aout]",
                 "-t", f"{duration:.3f}", "-c:a", "aac", "-b:a", "192k", bg_new_path],
                capture_output=True, text=True, errors="replace", timeout=300,
            )
            if merge_proc.returncode != 0 or not os.path.exists(bg_new_path) or os.path.getsize(bg_new_path) == 0:
                return _fail("Arka plan sesi kaydedilemedi: " + (merge_proc.stderr or "")[-400:])

            out_dur = _video_duration(ffmpeg_bin, bg_new_path)
            if duration > 0 and (out_dur <= 0 or abs(out_dur - duration) / duration > 0.02):
                return _fail(
                    "Üretilen arka plan sesi süresi videoyla uyuşmuyor "
                    f"(video: {duration:.1f} sn, üretilen: {out_dur:.1f} sn)"
                )

            # Her sey basarili: eski dosyanin yerine ATOMIK olarak yeni dosyayi koy.
            os.replace(bg_new_path, bg_path)
    except subprocess.TimeoutExpired:
        return _fail("Ayrıştırma zaman aşımına uğradı")
    except Exception as exc:
        return _fail("Ayrıştırma hatası: " + str(exc))
    finally:
        if os.path.exists(bg_new_path):
            try:
                os.remove(bg_new_path)
            except OSError:
                pass

    sig = _cue_signature(cues)
    conn = get_db()
    conn.execute(
        "UPDATE videos SET bg_file = ?, bg_method = ?, bg_cue_sig = ?, bg_error = NULL WHERE id = ?",
        (bg_filename, method, sig, video_id),
    )
    conn.commit()
    conn.close()
    _update_separation_state(video_id, stage="Tamamlandı", progress=100.0, eta_sec=0)
    return True, method, "ok" + fallback_note


def _run_separation_job(video_id: int):
    ok, method, message = separate_audio(video_id)
    with SEPARATION_LOCK:
        st = dict(SEPARATION_STATE.get(video_id, {}))
        st.update({
            "status": "done" if ok else "error",
            "method": method or None,
            "message": message,
            "progress": 100.0 if ok else st.get("progress"),
            "stage": "Tamamlandı" if ok else "Hata",
            "eta_sec": 0 if ok else None,
        })
        SEPARATION_STATE[video_id] = st


@app.post("/api/video/{video_id}/separate")
def api_separate_start(video_id: int):
    conn = get_db()
    v = conn.execute("SELECT id FROM videos WHERE id = ?", (video_id,)).fetchone()
    conn.close()
    if not v:
        return JSONResponse({"error": "Video bulunamadı"}, status_code=404)

    with SEPARATION_LOCK:
        current = SEPARATION_STATE.get(video_id)
        if current and current.get("status") == "running":
            return JSONResponse({"status": "running"})
        SEPARATION_STATE[video_id] = {
            "status": "running", "method": None, "message": "",
            "progress": 0.0, "stage": "Başlatılıyor", "eta_sec": None, "device": None,
        }

    threading.Thread(target=_run_separation_job, args=(video_id,), daemon=True).start()
    return JSONResponse({"status": "running"})


@app.get("/api/video/{video_id}/separation")
def api_separation_status(video_id: int):
    conn = get_db()
    v = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
    cues = conn.execute(
        "SELECT start_sec, end_sec FROM cues WHERE video_id = ? ORDER BY start_sec ASC", (video_id,)
    ).fetchall()
    conn.close()
    if not v:
        return JSONResponse({"error": "Video bulunamadı"}, status_code=404)

    with SEPARATION_LOCK:
        mem_state = dict(SEPARATION_STATE.get(video_id, {}))

    if mem_state.get("status") == "running":
        status, method, message = "running", mem_state.get("method"), mem_state.get("message", "")
    elif v["bg_file"]:
        # Ayrıştırmadan sonra repliklerin degisip degismedigini imzayla kontrol et.
        current_sig = _cue_signature(cues) if cues else None
        stored_sig = v["bg_cue_sig"] if "bg_cue_sig" in v.keys() else None
        if current_sig and stored_sig and current_sig != stored_sig:
            status = "stale"
        else:
            status = "done"
        method, message = v["bg_method"], "ok"
    elif mem_state.get("status") == "error":
        status, method, message = "error", mem_state.get("method"), mem_state.get("message", "")
    elif v["bg_error"] if "bg_error" in v.keys() else None:
        # Sunucu yeniden baslamis olabilir (bellek durumu kaybolur) — son hata DB'de kalir.
        status, method, message = "error", None, v["bg_error"]
    else:
        status, method, message = "none", None, ""

    return JSONResponse({
        "status": status,
        "method": method,
        "message": message,
        "demucs_available": _demucs_available(),
        "bg_url": f"/bg/{v['bg_file']}" if v["bg_file"] else None,
        "progress": mem_state.get("progress"),
        "stage": mem_state.get("stage"),
        "eta_sec": mem_state.get("eta_sec"),
        "device": mem_state.get("device"),
        "cue_count": len(cues),
    })


@app.post("/api/video/{video_id}/separation/delete")
def api_separation_delete(video_id: int):
    conn = get_db()
    v = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
    if not v:
        conn.close()
        return JSONResponse({"error": "Video bulunamadı"}, status_code=404)

    if v["bg_file"]:
        bg_path = os.path.join(BG_DIR, safe_filename(v["bg_file"]))
        if os.path.exists(bg_path):
            os.remove(bg_path)
        conn.execute(
            "UPDATE videos SET bg_file = NULL, bg_method = NULL, bg_cue_sig = NULL WHERE id = ?",
            (video_id,),
        )
        conn.commit()
    conn.close()

    with SEPARATION_LOCK:
        SEPARATION_STATE.pop(video_id, None)

    return JSONResponse({"ok": True})


@app.get("/bg/{filename}")
def bg_file_route(filename: str):
    safe = safe_filename(filename)
    path = os.path.join(BG_DIR, safe)
    if not os.path.exists(path):
        return JSONResponse({"error": "Dosya bulunamadı"}, status_code=404)
    return FileResponse(path)


def render_take(take_id: int):
    """Take icin ffmpeg montaji calistir. (ok: bool, message: str) dondurur."""
    conn = get_db()
    take = conn.execute("SELECT * FROM takes WHERE id = ?", (take_id,)).fetchone()
    if not take:
        conn.close()
        return False, "Take bulunamadı"

    # Yeniden montajda, islem basarisiz olursa kullanicinin elindeki calisan
    # videonun kaybolmamasi icin onceki ciktiyi hatirla.
    old_output = take["output_file"]

    video = conn.execute("SELECT * FROM videos WHERE id = ?", (take["video_id"],)).fetchone()
    if not video:
        conn.close()
        return False, "Video bulunamadı"

    recordings = conn.execute(
        """SELECT r.id AS rec_id, r.filename AS rec_filename, r.volume AS rec_volume,
                  r.mean_db AS rec_mean_db, r.peak_db AS rec_peak_db,
                  c.start_sec, c.end_sec
           FROM recordings r JOIN cues c ON c.id = r.cue_id
           WHERE r.take_id = ? ORDER BY c.start_sec ASC""",
        (take_id,),
    ).fetchall()
    conn.close()

    if not recordings:
        msg = "En az bir replik kaydedilmeli"
        _fail_take(take_id, old_output, msg)
        return False, msg

    video_path = os.path.join(UPLOADS_DIR, safe_filename(video["filename"]))
    if not os.path.exists(video_path):
        msg = "Video dosyası bulunamadı"
        _fail_take(take_id, old_output, msg)
        return False, msg

    # Legacy/bozuk veri savunmasi: _clean_cue_fields normal akista inf/NaN ve
    # asiri buyuk start/end degerlerini zaten reddeder, ama DB'ye dogrudan
    # enjekte edilmis eski/bozuk cue kayitlari yine de bulunabilir. ffmpeg
    # between(t,a,inf) ifadesini sessizce gecerli kabul ettigi icin boyle bir
    # replik sonsuz dongu ya da hataya degil, replik sesi pencere disina
    # tasarak "done" durumuna dusebilir. Filter_complex kurulmadan once acikca
    # reddet.
    for rec in recordings:
        s, e = rec["start_sec"], rec["end_sec"]
        if not (math.isfinite(s) and math.isfinite(e)) or s > _MAX_CUE_SEC or e > _MAX_CUE_SEC:
            msg = (
                f"Başlangıcı {s:.1f} sn olan repliğin zaman bilgisi geçersiz, "
                "montaj yapılamadı. Lütfen bu repliği düzenleyip tekrar deneyin."
            )
            print(
                f"[uyari] take={take_id} gecersiz cue zaman degeri (start={s}, end={e}), montaj reddedildi",
                file=sys.stderr,
            )
            _fail_take(take_id, old_output, msg)
            return False, msg

    _set_take_status(take_id, "rendering", old_output, None)

    # Bu noktadan sonra take DB'de "rendering" durumunda. Beklenmedik (yakalanmamis)
    # bir istisna (orn. OverflowError, disk/izin hatasi) burada patlarsa take sonsuza
    # dek "Montajlanıyor"da asili kalirdi (UI'da cikis yolu yok, kullanici take'i
    # silmek zorunda kalirdi). Bu yuzden gerisi tek bir try/except ile sarili:
    # her beklenmeyen hata _fail_take ile 'error' durumuna dusurulur.
    try:
        ffmpeg_bin = _ffmpeg_bin()
        has_audio = _video_has_audio(ffmpeg_bin, video_path)

        # keep_background=1 ve videonun ayrılmış arka plan dosyası varsa, replik
        # aralığında sessizlik yerine arka plan sesi korunur. Kapalıysa (varsayılan)
        # veya bg dosyası yoksa davranış tamamen eskisiyle aynı kalır (regresyon yok).
        bg_path = None
        if take["keep_background"] and video["bg_file"] and has_audio:
            candidate = os.path.join(BG_DIR, safe_filename(video["bg_file"]))
            if os.path.exists(candidate):
                bg_path = candidate

        input_args = ["-i", video_path]
        input_index = 1
        bg_index = None
        if bg_path:
            base_index = 0
            input_args += ["-i", bg_path]
            bg_index = input_index
            input_index += 1
        elif has_audio:
            base_index = 0
        else:
            base_index = input_index
            input_args += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
            input_index += 1

        rec_indices = []
        for rec in recordings:
            rec_path = os.path.join(REC_DIR, safe_filename(rec["rec_filename"]))
            input_args += ["-i", rec_path]
            rec_indices.append(input_index)
            input_index += 1

        # Arka plan / orijinal ses kazanci (yuzde 0-150). 100 iken gain_f bos string
        # kalir, yani uretilen filter_complex bugunkuyle birebir ayni olur (regresyon yok).
        bg_gain = max(0, min(150, int(take["bg_volume"] if take["bg_volume"] is not None else 100))) / 100.0
        gain_f = "" if abs(bg_gain - 1.0) < 1e-6 else f"volume={bg_gain:.3f},"

        # Dublaj kayitlarinin master seviyesi (yuzde 0-200) ve otomatik esitleme
        # anahtari. Varsayilanlarda (dub_volume=100, auto_level=0) her iki deger
        # de notr kalir -> filter_complex bugunkuyle birebir ayni olur (regresyon yok).
        dub_master = max(0, min(200, int(take["dub_volume"] if take["dub_volume"] is not None else 100))) / 100.0
        auto_on = bool(take["auto_level"])
        auto_gains = _auto_level_gains(ffmpeg_bin, recordings) if auto_on else None

        aformat = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
        enable_expr = "+".join(f"between(t,{rec['start_sec']},{rec['end_sec']})" for rec in recordings)
        filter_parts = [f"[{base_index}:a]{aformat},{gain_f}volume=0:enable='{enable_expr}'[base]"]
        mix_labels = ["[base]"]
        if bg_index is not None:
            filter_parts.append(f"[{bg_index}:a]{aformat},{gain_f}volume=0:enable='not({enable_expr})'[bg]")
            mix_labels.append("[bg]")
        max_gain = max(1.0, bg_gain)  # bg/orijinal ses boost'u da limiter tetiklemeli
        for i, rec in enumerate(recordings, start=1):
            idx = rec_indices[i - 1]
            delay_ms = int(round(float(rec["start_sec"]) * 1000))
            g = _effective_dub_gain(dub_master, rec, auto_gains, i - 1)
            dgain_f = "" if abs(g - 1.0) < 1e-6 else f"volume={g:.3f},"
            filter_parts.append(f"[{idx}:a]{aformat},{dgain_f}adelay={delay_ms}|{delay_ms}[a{i}]")
            mix_labels.append(f"[a{i}]")
            if g > max_gain:
                max_gain = g
        mix_inputs = "".join(mix_labels)
        mix_chain = f"amix=inputs={len(mix_labels)}:duration=first:normalize=0"
        limiter_on = max_gain > 1.0 + 1e-6 and _has_alimiter(ffmpeg_bin)
        if limiter_on:
            # limit=0.85: AAC (192k) teslimatinda intersample overshoot nedeniyle
            # limiter kendi hedefine (PRE-AAC PCM'de olcum: limit=0.95 -> -0.4dB,
            # 0.89 -> -1.0dB, 0.85 -> -1.4dB) ulassa bile kodlanmis dosyada 0 dBFS'e
            # dokunan orneklerin tamamen onune gecilemiyor (bilinen kayipli kodek
            # karakteristigi). Olculen (take=2, bg=150%): histogram_0db orneği
            # limit=0.95'te 6462, 0.89'da 335, 0.85'te 38 (2.4M orneginin ~%0.0016'si)
            # - politika geregi 0.85'in altina inilmiyor, bu deger en dusuk kalinti.
            mix_chain += ",alimiter=limit=0.85:level=disabled:attack=5:release=50"
        filter_parts.append(f"{mix_inputs}{mix_chain}[aout]")
        filter_complex = ";".join(filter_parts)

        print(
            f"[montaj] take={take_id} bg={bg_gain:.2f} dub={dub_master:.2f} "
            f"auto={int(auto_on)} max_gain={max_gain:.3f} limiter={int(limiter_on)}",
            file=sys.stderr,
        )

        output_filename = f"{uuid.uuid4().hex}.mp4"
        output_path = os.path.join(OUTPUTS_DIR, output_filename)

        base_cmd = [ffmpeg_bin, "-y", "-nostdin", "-hide_banner", "-loglevel", "error"]
        base_cmd += input_args
        base_cmd += ["-filter_complex", filter_complex, "-map", "0:v"]
        tail_cmd = ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"]
        if not has_audio:
            tail_cmd += ["-shortest"]
        tail_cmd += [output_path]

        def _run(video_codec_args):
            cmd = base_cmd + video_codec_args + tail_cmd
            try:
                return subprocess.run(
                    cmd, capture_output=True, text=True, errors="replace", timeout=1800
                )
            except subprocess.TimeoutExpired:
                return None

        proc = _run(["-c:v", "copy"])
        if proc is None or proc.returncode != 0:
            proc = _run(["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"])

        if proc is None:
            msg = "Montaj zaman aşımına uğradı"
            _fail_take(take_id, old_output, msg)
            return False, msg

        if proc.returncode != 0 or not os.path.exists(output_path):
            msg = (proc.stderr or "Bilinmeyen ffmpeg hatası")[-500:]
            _fail_take(take_id, old_output, msg)
            return False, msg

        _set_take_status(take_id, "done", output_filename, None)

        conn = get_db()
        conn.execute(
            "UPDATE takes SET rendered_at = ?, rendered_rec_count = ? WHERE id = ?",
            (datetime.now().isoformat(timespec="seconds"), len(recordings), take_id),
        )
        conn.commit()
        conn.close()

        # Yeniden montajda eski cikti artik hicbir yerden referans edilmiyor; diskte
        # yetim kalmamasi icin sil. Silme basarisiz olsa bile (Windows'ta dosya
        # acikken PermissionError, ya da baska bir OSError) montaj basarili sayilir:
        # eski dosya diskte yetim kalir ama kullanici 'basarili montaj hata verdi'
        # sanip kafasi karismaz, ve suanki oynatma akisi kirilmaz.
        if old_output and old_output != output_filename:
            old_path = os.path.join(OUTPUTS_DIR, safe_filename(old_output))
            try:
                if os.path.exists(old_path):
                    os.remove(old_path)
                    print(f"[bakim] eski montaj silindi: {old_output}", file=sys.stderr)
            except OSError as exc:
                print(f"[uyari] eski montaj silinemedi ({old_output}): {exc}", file=sys.stderr)

        return True, "ok"
    except Exception as exc:
        msg = f"Beklenmeyen montaj hatası: {exc}"
        traceback.print_exc(file=sys.stderr)
        _fail_take(take_id, old_output, msg)
        return False, msg


def _clean_volume(raw, lo=0, hi=200):
    """Form'dan gelen bir ses seviyesini [lo, hi] araligina kis; gecersizse None."""
    if raw is None:
        return None
    try:
        return max(lo, min(hi, int(float(raw))))
    except (TypeError, ValueError, OverflowError):
        # int(float("inf")) TypeError/ValueError degil OverflowError firlatir.
        return None


def _clean_bg_volume(raw):
    """Form'dan gelen arka plan seviyesini 0-150 araligina kis; gecersizse None."""
    return _clean_volume(raw, 0, 150)


@app.post("/take/{take_id}/save")
def save_take(
    take_id: int,
    name: str = Form(...),
    keep_background: str = Form(None),
    bg_volume: str = Form(None),
):
    conn = get_db()
    take = conn.execute("SELECT * FROM takes WHERE id = ?", (take_id,)).fetchone()
    if not take:
        conn.close()
        return RedirectResponse(url="/?error=" + quote("Take bulunamadı"), status_code=303)

    video_id = take["video_id"]

    # Eszamanli tetikleme korumasi: rerender_take'teki ayni kontrol (bkz. orada
    # ki yorum). Zaten calisiyorsa reddet, yetim cikti dosyasi birikmesin.
    if take["status"] == "rendering":
        conn.close()
        msg = quote("Bu take zaten montajlanıyor; lütfen bitmesini bekleyin.")
        return RedirectResponse(url=f"/dub/{video_id}?take={take_id}&error={msg}", status_code=303)

    rec_count = conn.execute(
        "SELECT COUNT(*) AS c FROM recordings WHERE take_id = ?", (take_id,)
    ).fetchone()["c"]
    if rec_count == 0:
        conn.close()
        msg = quote("En az bir replik kaydedilmeli")
        return RedirectResponse(url=f"/dub/{video_id}?take={take_id}&error={msg}", status_code=303)

    clean_name = (name or "").strip()[:120] or f"Dublaj {take_id}"
    keep_bg = 1 if (keep_background or "").strip().lower() in ("1", "on", "true") else 0
    bg_vol = _clean_bg_volume(bg_volume)
    if bg_vol is None:
        # Alan gonderilmediyse (eski form) mevcut deger korunur - geriye uyumlu.
        conn.execute(
            "UPDATE takes SET name = ?, keep_background = ? WHERE id = ?",
            (clean_name, keep_bg, take_id),
        )
    else:
        conn.execute(
            "UPDATE takes SET name = ?, keep_background = ?, bg_volume = ? WHERE id = ?",
            (clean_name, keep_bg, bg_vol, take_id),
        )
    conn.commit()
    conn.close()

    ok, message = render_take(take_id)
    if ok:
        return RedirectResponse(url=f"/take/{take_id}", status_code=303)
    return RedirectResponse(
        url=f"/dub/{video_id}?take={take_id}&error=" + quote(message[:200]), status_code=303
    )


@app.post("/take/{take_id}/rerender")
async def rerender_take(take_id: int, request: Request):
    """Mevcut kayitlarla montaji yeniden calistir (ayarlar degistiyse gunceller).

    rec_volume_{cue_id} alanlari dinamik oldugu icin (kac replik oldugu onceden
    bilinmez) FastAPI'de tipli Form parametresi olarak tanimlanamaz; tum alanlar
    request.form() uzerinden okunur. HTTP sozlesmesi degismez (take.html'deki
    tek cagiran form ayni alan adlarini gonderiyor).
    """
    form = await request.form()
    bg_volume = form.get("bg_volume")
    keep_background = form.get("keep_background")
    dub_volume = form.get("dub_volume")
    auto_level = form.get("auto_level")

    conn = get_db()
    take = conn.execute("SELECT * FROM takes WHERE id = ?", (take_id,)).fetchone()
    if not take:
        conn.close()
        return RedirectResponse(url="/?error=" + quote("Take bulunamadı"), status_code=303)

    # Eszamanli tetikleme korumasi: ayni take icin iki montaj birden calisirsa
    # ikisi de kendi eski ciktisini yakalar, biri digerinin ciktisini referanssiz
    # birakir (outputs/ altinda yetim .mp4 birikir). Zaten calisiyorsa reddet.
    if take["status"] == "rendering":
        conn.close()
        msg = quote("Bu take zaten montajlanıyor; lütfen bitmesini bekleyin.")
        return RedirectResponse(url=f"/take/{take_id}?error={msg}", status_code=303)

    rec_count = conn.execute(
        "SELECT COUNT(*) AS c FROM recordings WHERE take_id = ?", (take_id,)
    ).fetchone()["c"]
    if rec_count == 0:
        conn.close()
        msg = quote("En az bir replik kaydedilmeli")
        return RedirectResponse(url=f"/take/{take_id}?error={msg}", status_code=303)

    keep_bg = 1 if (keep_background or "").strip().lower() in ("1", "on", "true") else 0
    auto_on = 1 if (auto_level or "").strip().lower() in ("1", "on", "true") else 0
    bg_vol = _clean_bg_volume(bg_volume)
    dub_vol = _clean_volume(dub_volume, 0, 200)

    set_clauses = ["keep_background = ?", "auto_level = ?"]
    params = [keep_bg, auto_on]
    if bg_vol is not None:
        set_clauses.append("bg_volume = ?")
        params.append(bg_vol)
    if dub_vol is not None:
        set_clauses.append("dub_volume = ?")
        params.append(dub_vol)
    params.append(take_id)
    conn.execute(f"UPDATE takes SET {', '.join(set_clauses)} WHERE id = ?", params)

    # rec_volume_{cue_id}: bilinmeyen/gecersiz cue_id'ler WHERE take_id = ? AND
    # cue_id = ? ile zaten hicbir satira eslesmedigi icin sessizce yok sayilir.
    for key, raw_val in form.multi_items():
        if not key.startswith("rec_volume_"):
            continue
        cue_id_str = key[len("rec_volume_"):]
        try:
            cue_id = int(cue_id_str)
        except (TypeError, ValueError):
            continue
        rec_vol = _clean_volume(raw_val, 0, 200)
        if rec_vol is None:
            continue
        conn.execute(
            "UPDATE recordings SET volume = ? WHERE take_id = ? AND cue_id = ?",
            (rec_vol, take_id, cue_id),
        )

    conn.commit()
    conn.close()

    # render_take senkron ve bloklayici (ffmpeg subprocess.run + olcum): async
    # route'ta dogrudan cagirilirsa event loop'u montaj boyunca (en kotu 1800 sn)
    # dondurur. Threadpool'a tasi, event loop bosta kalsin.
    ok, message = await run_in_threadpool(render_take, take_id)
    if ok:
        return RedirectResponse(url=f"/take/{take_id}", status_code=303)
    return RedirectResponse(
        url=f"/take/{take_id}?error=" + quote(message[:200]), status_code=303
    )


@app.get("/takes/{video_id}")
def takes_list(request: Request, video_id: int):
    conn = get_db()
    v = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
    if not v:
        conn.close()
        return RedirectResponse(url="/?error=" + quote("Video bulunamadı"), status_code=303)
    takes = conn.execute(
        "SELECT * FROM takes WHERE video_id = ? ORDER BY id DESC", (video_id,)
    ).fetchall()
    conn.close()
    return templates.TemplateResponse(request, "takes.html", {"video": v, "takes": takes})


@app.get("/take/{take_id}")
def take_detail(request: Request, take_id: int):
    conn = get_db()
    take = conn.execute("SELECT * FROM takes WHERE id = ?", (take_id,)).fetchone()
    if not take:
        conn.close()
        return RedirectResponse(url="/?error=" + quote("Take bulunamadı"), status_code=303)
    video = conn.execute("SELECT * FROM videos WHERE id = ?", (take["video_id"],)).fetchone()
    cues = conn.execute(
        "SELECT * FROM cues WHERE video_id = ? ORDER BY start_sec ASC", (take["video_id"],)
    ).fetchall()
    recs = conn.execute(
        "SELECT cue_id, filename, created_at, volume FROM recordings WHERE take_id = ?", (take_id,)
    ).fetchall()
    conn.close()

    # DIKKAT: burada cue_id INT anahtar olarak kullanilir (take.html:65 buna bagimli).
    # api_take() ayni sozlugu STR anahtarla uretir (dub.html buna bagimli) - kasitli
    # fark, degistirme (bkz. api_take() yorumu).
    recordings = {r["cue_id"]: f"/rec/{r['filename']}" for r in recs}
    rec_volumes = {r["cue_id"]: (r["volume"] if r["volume"] is not None else 100) for r in recs}

    # Son montajdan sonra kayit eklendi/degistirildi/SILINDI mi? (ISO zaman damgalari,
    # ikisi de datetime.now().isoformat(timespec="seconds") ile yazilir.)
    # rendered_at NULL ise (bu migration'dan onceki take'ler) guvenli varsayilan:
    # uyari gosterme.
    #
    # Zaman damgasi karsilastirmasi tek basina kayit SILINMESINI goremez (silinen
    # satirin created_at'i de onunla birlikte gider, kalanlarin created_at'i
    # degismez). Bu yuzden montaj anindaki kayit sayisini da (rendered_rec_count)
    # karsilastiriyoruz; sayi degistiyse (ekleme VEYA silme) uyari gosterilir.
    # rendered_rec_count NULL ise (bu migrasyondan onceki take) sadece eski
    # zaman damgasi kontrolune dusulur - geriye uyumlu, false-positive yok.
    recordings_changed = False
    if take["rendered_at"]:
        rendered_rec_count = take["rendered_rec_count"]
        if rendered_rec_count is not None and len(recs) != rendered_rec_count:
            recordings_changed = True
        elif recs:
            last_rec = max((r["created_at"] or "") for r in recs)
            recordings_changed = last_rec > take["rendered_at"]

    return templates.TemplateResponse(
        request,
        "take.html",
        {
            "take": take,
            "video": video,
            "cues": cues,
            "recordings": recordings,
            "rec_volumes": rec_volumes,
            "recordings_changed": recordings_changed,
            "error": request.query_params.get("error"),
        },
    )


@app.post("/take/{take_id}/delete")
def delete_take(take_id: int):
    conn = get_db()
    take = conn.execute("SELECT * FROM takes WHERE id = ?", (take_id,)).fetchone()
    if not take:
        conn.close()
        return RedirectResponse(url="/?error=" + quote("Take bulunamadı"), status_code=303)

    video_id = take["video_id"]
    recs = conn.execute("SELECT * FROM recordings WHERE take_id = ?", (take_id,)).fetchall()
    for r in recs:
        rec_path = os.path.join(REC_DIR, safe_filename(r["filename"]))
        if os.path.exists(rec_path):
            os.remove(rec_path)

    if take["output_file"]:
        out_path = os.path.join(OUTPUTS_DIR, safe_filename(take["output_file"]))
        if os.path.exists(out_path):
            os.remove(out_path)

    conn.execute("DELETE FROM recordings WHERE take_id = ?", (take_id,))
    conn.execute("DELETE FROM takes WHERE id = ?", (take_id,))
    conn.commit()
    conn.close()
    return RedirectResponse(url=f"/takes/{video_id}", status_code=303)


@app.get("/rec/{filename}")
def rec_file(filename: str):
    safe = safe_filename(filename)
    path = os.path.join(REC_DIR, safe)
    if not os.path.exists(path):
        return JSONResponse({"error": "Dosya bulunamadı"}, status_code=404)
    return FileResponse(path)


@app.get("/output/{filename}")
def output_file(filename: str):
    safe = safe_filename(filename)
    path = os.path.join(OUTPUTS_DIR, safe)
    if not os.path.exists(path):
        return JSONResponse({"error": "Dosya bulunamadı"}, status_code=404)
    return FileResponse(path)
