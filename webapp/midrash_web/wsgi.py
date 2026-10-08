import os
import pathlib
import sys

from django.core.wsgi import get_wsgi_application

# The same two paths manage.py puts on sys.path: a server started from wsgi.py
# never runs manage.py, and without the second it cannot import pipeline_core.
here = pathlib.Path(__file__).resolve().parent.parent
for p in (here, here.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "midrash_web.settings")
application = get_wsgi_application()
