import os
from flask import Flask, redirect, request, session, jsonify, render_template
from dotenv import load_dotenv
from spotify_client import SpotifyClient, TokenExpiredError
from pipeline import MusicPipeline

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app.secret_key = os.getenv("FLASK_SECRET_KEY", "sonigram-secret")

CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:5000/callback")

spotify = SpotifyClient(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI)


def get_valid_token():
    """Return a valid access token, refreshing it automatically if expired."""
    token = session.get("access_token")
    if not token:
        return None

    # Try a lightweight probe — if it fails with 401, refresh
    try:
        spotify.get("/me", token)
        return token
    except TokenExpiredError:
        refresh = session.get("refresh_token")
        if not refresh:
            return None
        try:
            token_data = spotify.refresh_token(refresh)
            session["access_token"] = token_data["access_token"]
            # Spotify sometimes issues a new refresh token too
            if "refresh_token" in token_data:
                session["refresh_token"] = token_data["refresh_token"]
            return session["access_token"]
        except Exception:
            session.clear()
            return None


@app.route("/")
def index():
    logged_in = "access_token" in session
    return render_template("index.html", logged_in=logged_in)


@app.route("/login")
def login():
    auth_url = spotify.get_auth_url()
    return redirect(auth_url)


@app.route("/callback")
def callback():
    code = request.args.get("code")
    if not code:
        return "Authorization failed.", 400
    token_data = spotify.exchange_code(code)
    session["access_token"] = token_data["access_token"]
    session["refresh_token"] = token_data.get("refresh_token")
    return redirect("/visualize")


@app.route("/visualize")
def visualize():
    if "access_token" not in session:
        return redirect("/")
    return render_template("visualize.html")


@app.route("/api/music-map")
def music_map():
    token = get_valid_token()
    if not token:
        # Token gone / unrefreshable — ask the frontend to re-login
        return jsonify({"error": "session_expired"}), 401

    pipeline = MusicPipeline(token, spotify)
    try:
        result = pipeline.run()
    except TokenExpiredError:
        session.clear()
        return jsonify({"error": "session_expired"}), 401

    return jsonify(result)


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
