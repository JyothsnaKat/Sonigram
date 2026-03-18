import requests
import base64
import urllib.parse


SCOPES = "user-top-read user-read-recently-played user-library-read"
AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
API_BASE = "https://api.spotify.com/v1"


class TokenExpiredError(Exception):
    """Raised when the Spotify access token has expired (401)."""
    pass


class SpotifyClient:
    def __init__(self, client_id, client_secret, redirect_uri):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

    def get_auth_url(self):
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": SCOPES,
        }
        return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"

    def exchange_code(self, code):
        credentials = self._b64_credentials()
        headers = {"Authorization": f"Basic {credentials}"}
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
        }
        r = requests.post(TOKEN_URL, headers=headers, data=data)
        r.raise_for_status()
        return r.json()

    def refresh_token(self, refresh_token):
        """Exchange a refresh token for a new access token."""
        credentials = self._b64_credentials()
        headers = {"Authorization": f"Basic {credentials}"}
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
        r = requests.post(TOKEN_URL, headers=headers, data=data)
        r.raise_for_status()
        return r.json()  # contains new access_token (and sometimes a new refresh_token)

    def get(self, endpoint, token, params=None):
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.get(f"{API_BASE}{endpoint}", headers=headers, params=params or {})
        if r.status_code == 401:
            raise TokenExpiredError("Access token expired")
        r.raise_for_status()
        return r.json()

    def get_top_tracks(self, token, limit=50, time_range="medium_term"):
        return self.get("/me/top/tracks", token, {"limit": limit, "time_range": time_range})

    def get_top_artists(self, token, limit=20, time_range="medium_term"):
        return self.get("/me/top/artists", token, {"limit": limit, "time_range": time_range})

    def get_audio_features(self, token, track_ids):
        ids = ",".join(track_ids)
        try:
            return self.get("/audio-features", token, {"ids": ids})
        except TokenExpiredError:
            raise  # let the caller handle re-auth
        except Exception:
            return {"audio_features": []}

    def get_recently_played(self, token, limit=50):
        return self.get("/me/player/recently-played", token, {"limit": limit})

    def _b64_credentials(self):
        return base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()
