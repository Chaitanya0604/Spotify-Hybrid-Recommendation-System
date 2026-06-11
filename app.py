import streamlit as st
import pandas as pd
from numpy import load
from scipy.sparse import load_npz

from content_based_filtering import content_recommendation
from collaborative_filtering import collaborative_recommendation
from hybrid_recommendations import HybridRecommenderSystem as HRS


# =============================================================================
# Constants
# =============================================================================

DATA_PATHS = {
    "cleaned_data":           "data/cleaned_data.csv",
    "transformed_data":       "data/transformed_data.npz",
    "track_ids":              "data/track_ids.npy",
    "filtered_data":          "data/collab_filtered_data.csv",
    "interaction_matrix":     "data/interaction_matrix.npz",
    "transformed_hybrid":     "data/transformed_hybrid_data.npz",
}

FILTERING_OPTIONS = [
    "Content-Based Filtering",
    "Collaborative Filtering",
    "Hybrid Recommender System",
]

# Weights for the hybrid recommender
WEIGHT_CONTENT_BASED  = 0.3
WEIGHT_COLLABORATIVE  = 0.7


# =============================================================================
# Data Loading
# =============================================================================

@st.cache_data
def load_all_data():
    """
    Loads all required data files once and caches them.
    Streamlit's @st.cache_data decorator ensures this function only
    runs on the first load — subsequent reruns reuse the cached result,
    keeping the app fast.
    """
    songs_data            = pd.read_csv(DATA_PATHS["cleaned_data"])
    transformed_data      = load_npz(DATA_PATHS["transformed_data"])
    track_ids             = load(DATA_PATHS["track_ids"], allow_pickle=True)
    filtered_data         = pd.read_csv(DATA_PATHS["filtered_data"])
    interaction_matrix    = load_npz(DATA_PATHS["interaction_matrix"])
    transformed_hybrid    = load_npz(DATA_PATHS["transformed_hybrid"])

    return songs_data, transformed_data, track_ids, filtered_data, interaction_matrix, transformed_hybrid


# =============================================================================
# Helpers
# =============================================================================

def song_exists(dataframe: pd.DataFrame, song_name: str, artist_name: str) -> bool:
    """Returns True if the song + artist combination exists in the given DataFrame."""
    return ((dataframe["name"] == song_name) & (dataframe["artist"] == artist_name)).any()


def display_recommendations(recommendations: pd.DataFrame) -> None:
    """
    Renders a list of recommended songs in the Streamlit UI.

    - Index 0  → shown as 'Currently Playing' (the seed/input song itself).
    - Index 1  → shown as 'Next Up'.
    - Index 2+ → shown as a numbered list.

    Each entry displays the song title, artist name, and an audio preview player.
    """
    for idx, row in recommendations.iterrows():
        # Capitalise each word in the song and artist name for display
        display_name   = row["name"].title()
        display_artist = row["artist"].title()
        preview_url    = row["spotify_preview_url"]

        if idx == 0:
            st.markdown("## Currently Playing")
            st.markdown(f"#### **{display_name}** by **{display_artist}**")
        elif idx == 1:
            st.markdown("### Next Up 🎵")
            st.markdown(f"#### {idx}. **{display_name}** by **{display_artist}**")
        else:
            st.markdown(f"#### {idx}. **{display_name}** by **{display_artist}**")

        st.audio(preview_url)
        st.write("---")


def show_not_found(song_name: str) -> None:
    """Displays a friendly error message when a song is not found in the database."""
    st.warning(f"Sorry, we couldn't find **{song_name.title()}** in our database. Please try another song.")


# =============================================================================
# Recommendation Logic
# =============================================================================

def run_content_based(song_name, artist_name, k, songs_data, transformed_data):
    """Fetches and displays content-based recommendations."""
    recommendations = content_recommendation(
        song_name=song_name,
        artist_name=artist_name,
        songs_data=songs_data,
        transformed_data=transformed_data,
        k=k,
    )
    display_recommendations(recommendations)


def run_collaborative(song_name, artist_name, k, track_ids, filtered_data, interaction_matrix):
    """Fetches and displays collaborative filtering recommendations."""
    recommendations = collaborative_recommendation(
        song_name=song_name,
        artist_name=artist_name,
        track_ids=track_ids,
        songs_data=filtered_data,
        interaction_matrix=interaction_matrix,
        k=k,
    )
    display_recommendations(recommendations)


def run_hybrid(song_name, artist_name, k, filtered_data, transformed_hybrid, track_ids, interaction_matrix):
    """Fetches and displays hybrid recommender system results."""
    recommender = HRS(
        song_name=song_name,
        artist_name=artist_name,
        number_of_recommendations=k,
        weight_content_based=WEIGHT_CONTENT_BASED,
        weight_collaborative=WEIGHT_COLLABORATIVE,
        songs_data=filtered_data,
        transformed_matrix=transformed_hybrid,
        track_ids=track_ids,
        interaction_matrix=interaction_matrix,
    )
    recommendations = recommender.give_recommendations()
    display_recommendations(recommendations)


# =============================================================================
# UI
# =============================================================================

def main():
    # ── Page title and description ───────────────────────────────────────────
    st.title("Welcome to the Spotify Song Recommender!")
    st.write("### Enter the name of a song and the recommender will suggest similar songs 🎵🎧")

    # ── Load data ────────────────────────────────────────────────────────────
    (
        songs_data,
        transformed_data,
        track_ids,
        filtered_data,
        interaction_matrix,
        transformed_hybrid,
    ) = load_all_data()

    # ── User inputs ──────────────────────────────────────────────────────────
    song_name   = st.text_input("Enter a song name:").strip().lower()
    artist_name = st.text_input("Enter the artist name:").strip().lower()

    k = st.selectbox(
        "How many recommendations do you want?",
        options=[5, 10, 15, 20],
        index=1,
    )

    filtering_type = st.selectbox(
        label="Select the type of filtering:",
        options=FILTERING_OPTIONS,
        index=2,  # Default to Hybrid
    )

    # ── Recommend button ─────────────────────────────────────────────────────
    if st.button("Get Recommendations"):

        # Validate that the user has entered something
        if not song_name or not artist_name:
            st.warning("Please enter both a song name and an artist name.")
            return

        st.write(f"Recommendations for **{song_name.title()}** by **{artist_name.title()}**")

        # ── Content-Based Filtering ──────────────────────────────────────────
        if filtering_type == "Content-Based Filtering":
            if song_exists(songs_data, song_name, artist_name):
                run_content_based(song_name, artist_name, k, songs_data, transformed_data)
            else:
                show_not_found(song_name)

        # ── Collaborative Filtering ──────────────────────────────────────────
        elif filtering_type == "Collaborative Filtering":
            if song_exists(filtered_data, song_name, artist_name):
                run_collaborative(song_name, artist_name, k, track_ids, filtered_data, interaction_matrix)
            else:
                show_not_found(song_name)

        # ── Hybrid Recommender System ────────────────────────────────────────
        elif filtering_type == "Hybrid Recommender System":
            if song_exists(filtered_data, song_name, artist_name):
                run_hybrid(song_name, artist_name, k, filtered_data, transformed_hybrid, track_ids, interaction_matrix)
            else:
                show_not_found(song_name)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    main()