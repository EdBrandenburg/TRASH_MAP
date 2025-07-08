# app/__init__.py

import sqlite3
from datetime import timedelta
from flask import Flask, g, current_app
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_mail import Mail
from flask_babel import Babel
from flask import request, session

# — Extensions — #
db    = SQLAlchemy()
login = LoginManager()
mail  = Mail() 
babel = Babel()

# — Application factory — #
def create_app():
    app = Flask(__name__)
    
    # 1) Load your normal config
    app.config.from_object('app.config.Config')

    # 2) Make sessions permanent by default and extend "remember me"
    app.config.setdefault('SESSION_PERMANENT', True)
    # how long the session (and remember‐me cookie) should last:
    app.config.setdefault('PERMANENT_SESSION_LIFETIME', timedelta(days=30))
    # specifically for Flask-Login remember cookie
    app.config.setdefault('REMEMBER_COOKIE_DURATION', timedelta(days=30))

    def get_locale():
        lang = request.args.get('lang')
        if lang in app.config['BABEL_SUPPORTED_LOCALES']:
            session['lang'] = lang
        return session.get(
            'lang',
            request.accept_languages.best_match(
                app.config['BABEL_SUPPORTED_LOCALES']
            )
        )

    # Init extensions
    db.init_app(app)
    login.init_app(app)
    mail.init_app(app)
    # Pass get_locale into Babel here
    babel.init_app(app, locale_selector=get_locale)
    login.login_view = 'main.login'

    @app.context_processor
    def inject_locale():
        return {'current_locale': get_locale()}

    # User loader
    from app.models import User
    @login.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Teardown for direct sqlite connections
    @app.teardown_appcontext
    def close_connection(exception):
        sqlite_db = getattr(g, '_database', None)
        if sqlite_db is not None:
            sqlite_db.close()

    # Register blueprints
    from app.routes import main_bp, profile_bp, quiz_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(profile_bp)
    app.register_blueprint(quiz_bp)

    return app

# — Helper to get direct sqlite3 cursor if you need it — #
def get_db():
    db_conn = getattr(g, '_database', None)
    if db_conn is None:
        # Assumes SQLALCHEMY_DATABASE_URI is a sqlite filepath
        uri = current_app.config['SQLALCHEMY_DATABASE_URI']
        db_conn = g._database = sqlite3.connect(uri)
        db_conn.row_factory = sqlite3.Row
    return db_conn