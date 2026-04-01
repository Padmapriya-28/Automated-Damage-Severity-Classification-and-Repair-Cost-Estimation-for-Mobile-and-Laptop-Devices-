import os

from flask import Flask, send_from_directory

from api.routes import api_bp
from utils.logger import setup_logging


def create_app() -> Flask:
    setup_logging()

    # Support both entrypoints: project-root main.py and backend/main.py.
    base_dir = os.path.dirname(__file__)
    frontend_candidates = [
        os.path.abspath(os.path.join(base_dir, "frontend")),
        os.path.abspath(os.path.join(base_dir, "..", "frontend")),
    ]
    frontend_dir = next((path for path in frontend_candidates if os.path.isdir(path)), frontend_candidates[0])

    app = Flask(__name__, static_folder=frontend_dir, static_url_path="/static")

    app.register_blueprint(api_bp, url_prefix="/api")

    @app.get("/")
    def index():
        return send_from_directory(frontend_dir, "index.html")

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
