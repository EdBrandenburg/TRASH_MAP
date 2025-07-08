import os
from datetime import timedelta

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-very-secret-key'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///../Database/database_factory.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Chemin vers la base SQLite pour le module explorer
    DATABASE = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'Database/database_factory.db')

    # Babel translation settings
    BABEL_DEFAULT_LOCALE    = 'fr'
    BABEL_SUPPORTED_LOCALES = ['fr', 'en']
    BABEL_TRANSLATION_DIRECTORIES = 'app/translations'

    # Mail
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USERNAME = 'trashmapflaskemailer@gmail.com'
    MAIL_PASSWORD = 'xtbk quwl kfob ldef'
    MAIL_DEFAULT_SENDER = ('Trash Map', 'trashmapflaskemailer@gmail.com')

    REMEMBER_COOKIE_DURATION = timedelta(days=30)
    SESSION_PERMANENT = True
    PERMANENT_SESSION_LIFETIME = timedelta(days=30)

    # Upload d'avatar
    AVATAR_UPLOAD_FOLDER = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'static/avatars'
    )

    UPLOAD_FOLDER = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'static/uploads'
    )
    