"""Settings for the cluster explorer.

Deliberately minimal: no database, no auth, no static pipeline. The app reads the
embedding store from disk and writes each run's downloadable outputs to RUNS_DIR.
"""
import os
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

# Locally nothing needs setting. On the server (behind nginx, under /midrash/,
# password-protected there) the service sets these in its environment.
SECRET_KEY = os.environ.get("MIDRASH_SECRET_KEY", "midrash-explorer-not-a-secret")
DEBUG = os.environ.get("MIDRASH_DEBUG", "1") == "1"
ALLOWED_HOSTS = ["*"]
FORCE_SCRIPT_NAME = os.environ.get("MIDRASH_SCRIPT_NAME") or None

INSTALLED_APPS = ["django.contrib.staticfiles", "explorer"]
MIDDLEWARE = ["django.middleware.common.CommonMiddleware"]

ROOT_URLCONF = "midrash_web.urls"
WSGI_APPLICATION = "midrash_web.wsgi.application"

# Django caches templates in memory even under DEBUG, so a server started with
# --noreload keeps serving the template it read at startup. Load them directly.
TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "DIRS": [],
    "OPTIONS": {
        "context_processors": [],
        "loaders": ["django.template.loaders.app_directories.Loader"],
    },
}]

DATABASES = {}
STATIC_URL = "static/"
USE_TZ = True

# Where a run's CSV and HTML outputs are written for download.
RUNS_DIR = pathlib.Path(os.environ.get("MIDRASH_RUNS_DIR", BASE_DIR / "_runs"))

# The most memory one run may be expected to use, in MB (see views._estimate_mb).
# None locally; on the shared server it keeps a run inside the service's memory
# limit, so a large selection is refused up front instead of being killed.
MAX_RUN_MB = int(os.environ["MIDRASH_MAX_RUN_MB"]) if os.environ.get("MIDRASH_MAX_RUN_MB") else None
