# Static root is absolute path to the directory where static files are collected
STATIC_ROOT = os.path.join(BASE_DIR, 'django_ace/static')
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'ncc-oj',
        'USER': 'root',
        'PASSWORD': 'nccoj@123',
        'HOST': '127.0.0.1',
        'OPTIONS': {
            'charset': 'utf8mb4'
        },
    }
}
MEZON_AUTH_CLIENT_ID = 'MEZON_AUTH_CLIENT_ID'
MEZON_AUTH_CLIENT_SECRET = 'MEZON_AUTH_CLIENT_SECRET'
MEZON_AUTH_REDIRECT_URL ='http://localhost:8000/accounts/auth/callback'
MEZON_AUTH_URL = 'https://oauth2.mezon.ai'

CELERY_BROKER_URL = 'redis://localhost:6379'
CELERY_RESULT_BACKEND = 'redis://localhost:6379'
# Bridged configuration
BRIDGED_JUDGE_ADDRESS = [('localhost', 9999)]
BRIDGED_JUDGE_PROXIES = None
BRIDGED_DJANGO_ADDRESS = [('localhost', 9998)]
BRIDGED_DJANGO_CONNECT = None

DJANGO_CELERY_BEAT_TZ_AWARE = False
USE_TZ = True