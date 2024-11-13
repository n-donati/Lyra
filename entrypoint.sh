#!/bin/bash

export PORT=${PORT:-8000}

python manage.py collectstatic --noinput
python manage.py makemigrations app
python manage.py makemigrations
python manage.py migrate

# Simplified gunicorn configuration
gunicorn rinna.wsgi:application \
    --bind 0.0.0.0:$PORT \
    --workers 2 \
    --threads 2 \
    --timeout 120 \
    --log-level debug \
    --capture-output