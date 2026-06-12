import streamlit as st
from content_based_filtering import content_recommendation
from scipy.sparse import load_npz
import pandas as pd
from collaborative_filtering import collaborative_recommendation
from numpy import load
from hybrid_recommendations import HybridRecommenderSystem


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

# Load the main songs dataset (50K songs with audio features)
cleaned_data_path = "data/cleaned_data.csv"
st.session_state.songs_data = pd.read_csv(cleaned_data_path)

# Load the content-based transformed feature matrix (TF-IDF / scaled features)
transformed_data_path = "data/transformed_data.npz"
st.session_state.transformed_data = load_npz(transformed_data_path)

# Load track IDs used to map matrix indices to songs
track_ids_path = "data/track_ids.npy"
st.session_state.track_ids = load(track_ids_path, allow_pickle=True)

# Load the filtered songs dataset (30K songs with listening history — used by collaborative & hybrid)
filtered_data_path = "data/collab_filtered_data.csv"
st.session_state.filtered_data = pd.read_csv(filtered_data_path)

# Load the user-song interaction matrix (sparse: users × songs)
interaction_matrix_path = "data/interaction_matrix.npz"
st.session_state.interaction_matrix = load_npz(interaction_matrix_path)

# Load the transformed feature matrix for hybrid (built on filtered 30K songs)
transformed_hybrid_data_path = "data/transformed_hybrid_data.npz"
st.session_state.transformed_hybrid_data = load_npz(transformed_hybrid_data_path)


# ─────────────────────────────────────────────
# UI — HEADER & INPUTS
# ─────────────────────────────────────────────

# App title
st.title('Welcome to the Spotify Song Recommender!')

# Instructions
st.write('### Enter the name of a song and the recommender will suggest similar songs 🎵🎧')

# Song name input
song_name = st.text_input('Enter a song name:')
st.write('You entered:', song_name)

# Artist name input
artist_name = st.text_input('Enter the artist name:')
st.write('You entered:', artist_name)

# Normalize inputs to lowercase for consistent matching
song_name = song_name.lower()
artist_name = artist_name.lower()

# Number of recommendations to return
k = st.selectbox('How many recommendations do you want?', [5, 10, 15, 20], index=1)


# ─────────────────────────────────────────────
# COLD START DETECTION
# Check if the song exists in the filtered (30K) dataset
# If yes → all three modes available + diversity slider
# If no  → cold-start song, only Content-Based is available
# ─────────────────────────────────────────────

if ((st.session_state.filtered_data["name"] == song_name) & (st.session_state.filtered_data["artist"] == artist_name)).any():

    # Song has listening history → unlock all filtering modes
    filtering_type = st.selectbox(label='Select the type of filtering:',
                                  options=['Content-Based Filtering',
                                           'Collaborative Filtering',
                                           'Hybrid Recommender System'],
                                  index=2)  # Default to Hybrid

    # Diversity slider — controls the content vs collaborative weight balance
    # Higher diversity → more content-based (explore by sound)
    # Lower diversity  → more collaborative (follow listening patterns)
    diversity = st.slider(label="Diversity in Recommendations",
                          min_value=1,
                          max_value=10,
                          value=5,
                          step=1)

    # Convert diversity to content-based weight
    # diversity=1  → content_weight=0.9 (mostly collaborative)
    # diversity=10 → content_weight=0.0 (fully collaborative)
    content_based_weight = 1 - (diversity / 10)

else:
    # Cold-start song → no listening history, only Content-Based is possible
    filtering_type = st.selectbox(label='Select the type of filtering:',
                                  options=['Content-Based Filtering'])


# ─────────────────────────────────────────────
# CONTENT-BASED FILTERING
# Runs on the full 50K songs dataset
# Recommends based on audio features & tags similarity
# ─────────────────────────────────────────────

if filtering_type == 'Content-Based Filtering':
    if st.button('Get Recommendations'):

        # Check if song exists in the full 50K dataset
        if ((st.session_state.songs_data["name"] == song_name) & (st.session_state.songs_data['artist'] == artist_name)).any():
            st.write('Recommendations for', f"**{song_name}** by **{artist_name}**")

            # Get content-based recommendations
            recommendations = content_recommendation(song_name=song_name,
                                                     artist_name=artist_name,
                                                     songs_data=st.session_state.songs_data,
                                                     transformed_data=st.session_state.transformed_data,
                                                     k=k)

            # Display each recommendation with Spotify preview
            for ind, recommendation in recommendations.iterrows():
                song_name = recommendation['name'].title()
                artist_name = recommendation['artist'].title()

                if ind == 0:
                    # First row is the input song itself — show as currently playing
                    st.markdown("## Currently Playing")
                    st.markdown(f"#### **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                elif ind == 1:
                    # First recommendation — highlight as next up
                    st.markdown("### Next Up 🎵")
                    st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                else:
                    # Remaining recommendations
                    st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
        else:
            st.write(f"Sorry, we couldn't find {song_name} in our database. Please try another song.")


# ─────────────────────────────────────────────
# COLLABORATIVE FILTERING
# Runs on the filtered 30K songs dataset
# Recommends based on user listening patterns (item-item similarity)
# ─────────────────────────────────────────────

elif filtering_type == 'Collaborative Filtering':
    if st.button('Get Recommendations'):

        # Check if song exists in the filtered 30K dataset
        if ((st.session_state.filtered_data["name"] == song_name) & (st.session_state.filtered_data["artist"] == artist_name)).any():
            st.write('Recommendations for', f"**{song_name}** by **{artist_name}**")

            # Get collaborative recommendations
            recommendations = collaborative_recommendation(song_name=song_name,
                                                           artist_name=artist_name,
                                                           track_ids=st.session_state.track_ids,
                                                           songs_data=st.session_state.filtered_data,
                                                           interaction_matrix=st.session_state.interaction_matrix,
                                                           k=k)

            # Display each recommendation with Spotify preview
            for ind, recommendation in recommendations.iterrows():
                song_name = recommendation['name'].title()
                artist_name = recommendation['artist'].title()

                if ind == 0:
                    st.markdown("## Currently Playing")
                    st.markdown(f"#### **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                elif ind == 1:
                    st.markdown("### Next Up 🎵")
                    st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                else:
                    st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
        else:
            st.write(f"Sorry, we couldn't find {song_name} in our database. Please try another song.")


# ─────────────────────────────────────────────
# HYBRID RECOMMENDER SYSTEM
# Combines Content-Based + Collaborative scores
# Weight controlled by the diversity slider
# final_score = content_weight × content_norm + (1 - content_weight) × collab_norm
# ─────────────────────────────────────────────

elif filtering_type == "Hybrid Recommender System":
    if st.button('Get Recommendations'):

        # Check if song exists in the filtered 30K dataset
        if ((st.session_state.filtered_data["name"] == song_name) & (st.session_state.filtered_data["artist"] == artist_name)).any():
            st.write('Recommendations for', f"**{song_name}** by **{artist_name}**")

            # Initialise the hybrid recommender with k and the user-controlled content weight
            recommender = HybridRecommenderSystem(
                number_of_recommendations=k,
                weight_content_based=content_based_weight
            )

            # Get hybrid recommendations
            recommendations = recommender.give_recommendations(
                song_name=song_name,
                artist_name=artist_name,
                songs_data=st.session_state.filtered_data,
                transformed_matrix=st.session_state.transformed_hybrid_data,
                track_ids=st.session_state.track_ids,
                interaction_matrix=st.session_state.interaction_matrix
            )

            # Display each recommendation with Spotify preview
            for ind, recommendation in recommendations.iterrows():
                song_name = recommendation['name'].title()
                artist_name = recommendation['artist'].title()

                if ind == 0:
                    st.markdown("## Currently Playing")
                    st.markdown(f"#### **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                elif ind == 1:
                    st.markdown("### Next Up 🎵")
                    st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
                else:
                    st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')
        else:
            st.write(f"Sorry, we couldn't find {song_name} in our database. Please try another song.")