#!/usr/bin/env python
"""Run the cluster explorer:  python manage.py runserver"""
import os
import pathlib
import sys


def main():
    here = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(here))            # webapp/  — midrash_web, explorer
    sys.path.insert(0, str(here.parent))     # Midrash/ — pipeline_core, cluster_visualization_report
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "midrash_web.settings")
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
