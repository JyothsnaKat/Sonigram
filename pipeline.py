import numpy as np
import hashlib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

AUDIO_FEATURE_KEYS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]

# Normalisation ranges for raw Spotify values → 0..1
FEATURE_RANGES = {
    "danceability":     (0.0,   1.0),
    "energy":           (0.0,   1.0),
    "loudness":         (-60.0, 0.0),
    "speechiness":      (0.0,   1.0),
    "acousticness":     (0.0,   1.0),
    "instrumentalness": (0.0,   1.0),
    "liveness":         (0.0,   1.0),
    "valence":          (0.0,   1.0),
    "tempo":            (50.0,  200.0),
}

# ── Dynamic cluster labeller ────────────────────────────────────────────────
# Rules checked in priority order; first match whose label is unused wins.
# Empty conditions = always matches. Multiple unique catch-alls at the end
# guarantee every cluster gets a distinct name — duplicates are impossible.
CLUSTER_RULES = [
    ("Dance Floor",       "#FFA07A", {"danceability": (0.65, 1.0), "energy": (0.55, 1.0)}),
    ("Hype Mode",         "#FFD166", {"tempo": (0.65, 1.0),        "energy": (0.65, 1.0)}),
    ("Energetic Anthems", "#FF6B6B", {"energy": (0.65, 1.0),       "valence": (0.45, 1.0)}),
    ("Feel-Good Hits",    "#06D6A0", {"valence": (0.65, 1.0),      "danceability": (0.50, 1.0)}),
    ("Deep Focus",        "#45B7D1", {"instrumentalness": (0.25, 1.0)}),
    ("Acoustic Soul",     "#98D8C8", {"acousticness": (0.55, 1.0)}),
    ("Sad Bangers",       "#9B8EC4", {"valence": (0.0,  0.35),     "energy": (0.55, 1.0)}),
    ("Melancholy",        "#7B9EA6", {"valence": (0.0,  0.35),     "energy": (0.0,  0.55)}),
    ("Chill Vibes",       "#4ECDC4", {"energy": (0.0,  0.45)}),
    ("Emotional Journey", "#C4A882", {"valence": (0.0,  0.50)}),
    ("Late Night Drive",  "#A78BFA", {"energy": (0.40, 0.70),      "valence": (0.30, 0.65)}),
    # Unique catch-alls — always match, each a different name
    ("Eclectic Mix",      "#8888AA", {}),
    ("Hidden Gems",       "#AA8866", {}),
    ("The Deep End",      "#668899", {}),
    ("Wildcard",          "#997766", {}),
    ("Off the Map",       "#669977", {}),
]


def normalise_features(raw_row):
    """Convert raw Spotify feature values to 0..1 for labelling logic."""
    normed = {}
    for i, k in enumerate(AUDIO_FEATURE_KEYS):
        lo, hi = FEATURE_RANGES[k]
        val = raw_row[i]
        normed[k] = max(0.0, min(1.0, (val - lo) / (hi - lo)))
    return normed


def label_cluster(centroid_normed):
    """Pick the best label for a cluster given its normalised centroid."""
    for label, color, conditions in CLUSTER_RULES:
        if all(lo <= centroid_normed[f] <= hi for f, (lo, hi) in conditions.items()):
            return label, color
    return "Mixed Vibes", "#888899"


class MusicPipeline:
    def __init__(self, token, spotify_client):
        self.token = token
        self.spotify = spotify_client

    def run(self):
        tracks = self._fetch_tracks()
        if not tracks:
            return {"error": "No tracks found. Make sure you have listening history."}

        track_ids = [t["id"] for t in tracks]
        features_data = self.spotify.get_audio_features(self.token, track_ids[:100])

        track_map = {t["id"]: t for t in tracks}
        feature_rows = []   # raw values (for ML pipeline)
        valid_tracks = []

        audio_features = features_data.get("audio_features", [])
        has_real_features = any(f is not None for f in audio_features)

        if has_real_features:
            for f in audio_features:
                if f is None:
                    continue
                tid = f["id"]
                if tid not in track_map:
                    continue
                row = [f.get(k, 0) or 0 for k in AUDIO_FEATURE_KEYS]
                feature_rows.append(row)
                valid_tracks.append(track_map[tid])
        else:
            # Fallback: estimate from metadata when audio-features is blocked
            for t in tracks:
                tid = t["id"]
                h = int(hashlib.md5(tid.encode()).hexdigest(), 16)

                def pseudo(offset, lo=0.2, hi=0.8):
                    return lo + ((h >> offset) & 0xFF) / 255.0 * (hi - lo)

                popularity = t.get("popularity", 50) / 100.0
                duration_ms = t.get("duration_ms", 210000)
                duration_factor = min(duration_ms / 300000, 1.0)

                # Raw values in their natural scales
                row = [
                    min(1.0, popularity * 0.6 + pseudo(0,  0.2, 0.5)),     # danceability 0-1
                    min(1.0, popularity * 0.5 + pseudo(8,  0.1, 0.6)),     # energy 0-1
                    pseudo(16, -14, -2),                                     # loudness dB
                    pseudo(24, 0.02, 0.25),                                  # speechiness 0-1
                    max(0.0, 0.5 - popularity * 0.3 + pseudo(32, 0, 0.4)), # acousticness 0-1
                    max(0.0, duration_factor * 0.3 + pseudo(40, 0, 0.3)),  # instrumentalness 0-1
                    pseudo(48, 0.05, 0.4),                                   # liveness 0-1
                    min(1.0, popularity * 0.4 + pseudo(56, 0.1, 0.6)),     # valence 0-1
                    pseudo(64, 80, 160),                                     # tempo BPM
                ]
                feature_rows.append(row)
                valid_tracks.append(t)

        if len(feature_rows) < 5:
            return {"error": "Not enough track data to generate a map."}

        X = np.array(feature_rows)

        # Normalise to 0-1 before ML (so loudness/tempo don't dominate)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA → 2D
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_scaled)

        # K-Means
        n_clusters = min(5, len(valid_tracks))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_ids = kmeans.fit_predict(X_scaled)

        # ── Dynamic label assignment per cluster ──────────────────────────
        # Sort clusters by size descending so dominant clusters get first pick
        cluster_sizes = {cid: int((cluster_ids == cid).sum()) for cid in range(n_clusters)}
        sorted_cids = sorted(range(n_clusters), key=lambda c: -cluster_sizes[c])

        cluster_labels = {}
        cluster_colors = {}
        used_labels = set()

        for cid in sorted_cids:
            mask = cluster_ids == cid
            centroid_raw = X[mask].mean(axis=0)
            centroid_normed = normalise_features(centroid_raw)

            # Find first matching rule whose label hasn't been used yet
            label, color = None, None
            for rule_label, rule_color, conditions in CLUSTER_RULES:
                if rule_label in used_labels:
                    continue
                if all(lo <= centroid_normed[f] <= hi for f, (lo, hi) in conditions.items()):
                    label, color = rule_label, rule_color
                    break

            # Safety net: append cluster index to make truly unique if needed
            if label is None or label in used_labels:
                label = f"Cluster {cid + 1}"
                color = "#777788"

            used_labels.add(label)
            cluster_labels[cid] = label
            cluster_colors[cid] = color

        # ── Build per-track output ────────────────────────────────────────
        points = []
        for i, track in enumerate(valid_tracks):
            cid = int(cluster_ids[i])
            artists = [a["name"] for a in track.get("artists", [])]
            album_img = ""
            images = track.get("album", {}).get("images", [])
            if images:
                album_img = images[-1]["url"]

            # Normalised features for display (0-1)
            normed = normalise_features(feature_rows[i])

            points.append({
                "id": track["id"],
                "name": track["name"],
                "artists": artists,
                "album": track.get("album", {}).get("name", ""),
                "album_img": album_img,
                "x": float(X_2d[i][0]),
                "y": float(X_2d[i][1]),
                "cluster": cid,
                "cluster_label": cluster_labels[cid],
                "cluster_color": cluster_colors[cid],
                "features": {k: round(float(v), 3) for k, v in normed.items()},
                "popularity": track.get("popularity", 0),
            })

        summary = self._build_personality(feature_rows, cluster_ids, n_clusters, cluster_labels)

        top_artists_data = self.spotify.get_top_artists(self.token, limit=5)
        top_artists = [a["name"] for a in top_artists_data.get("items", [])]

        genres = []
        for a in top_artists_data.get("items", []):
            genres.extend(a.get("genres", []))
        genre_counts = {}
        for g in genres:
            genre_counts[g] = genre_counts.get(g, 0) + 1
        top_genres = sorted(genre_counts.items(), key=lambda x: -x[1])[:12]

        return {
            "points": points,
            "summary": summary,
            "top_artists": top_artists,
            "top_genres": [{"genre": g, "count": c} for g, c in top_genres],
            "explained_variance": [float(v) for v in pca.explained_variance_ratio_],
            "n_tracks": len(points),
            "estimated_features": not has_real_features,
        }

    def _fetch_tracks(self):
        seen = set()
        tracks = []
        for time_range in ["short_term", "medium_term", "long_term"]:
            data = self.spotify.get_top_tracks(self.token, limit=50, time_range=time_range)
            for t in data.get("items", []):
                if t["id"] not in seen:
                    seen.add(t["id"])
                    tracks.append(t)
        return tracks

    def _build_personality(self, feature_rows, cluster_ids, n_clusters, cluster_labels):
        X = np.array(feature_rows)
        avg_raw = X.mean(axis=0)
        normed = normalise_features(avg_raw)

        traits = []
        if normed["energy"] > 0.70:       traits.append("High Energy")
        elif normed["energy"] < 0.35:     traits.append("Mellow")
        if normed["danceability"] > 0.68: traits.append("Dance Lover")
        if normed["acousticness"] > 0.50: traits.append("Acoustic Soul")
        if normed["instrumentalness"] > 0.30: traits.append("Instrumental Explorer")
        if normed["valence"] > 0.65:      traits.append("Optimist")
        elif normed["valence"] < 0.30:    traits.append("Introspective")
        if normed["speechiness"] > 0.18:  traits.append("Lyric Devotee")
        if normed["tempo"] > 0.65:        traits.append("Fast-Paced Listener")
        if not traits:                     traits.append("Eclectic Listener")

        cluster_sizes = np.bincount(cluster_ids, minlength=n_clusters)
        dominant = int(cluster_sizes.argmax())

        return {
            "traits": traits,
            "dominant_cluster": dominant,
            "dominant_label": cluster_labels.get(dominant, "Eclectic"),
            "avg_features": {k: round(float(v), 3) for k, v in normed.items()},
        }
