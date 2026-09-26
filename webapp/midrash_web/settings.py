"""Settings for the cluster explorer.

Deliberately minimal: no database, no auth, no static pipeline. The app reads the
embedding store from disk and writes each run's downloadable outputs to RUNS_DIR.
"""
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

SECRET_KEY = "midrash-explorer-not-a-secret"   # nothing here is user data or private
DEBUG = True
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = ["django.contrib.staticfiles", "explorer"]
MIDDLEWARE = ["django.middleware.common.CommonMiddleware"]

ROOT_URLCONF = "midrash_web.urls"
WSGI_APPLICATION = "midrash_web.wsgi.application"

TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "DIRS": [],
    "APP_DIRS": True,
    "OPTIONS": {"context_processors": []},
}]

DATABASES = {}
STATIC_URL = "static/"
USE_TZ = True

# Where a run's CSV and HTML outputs are written for download.
RUNS_DIR = BASE_DIR / "_runs"
