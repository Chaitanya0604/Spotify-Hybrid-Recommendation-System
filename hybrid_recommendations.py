import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# HybridRecommenderSystem
# =============================================================================
# A "hybrid" recommender combines two different recommendation strategies:
#
#   1. Content-Based Filtering:
#      Recommends songs that are *sonically/musically similar* to the input song.
#      It looks at audio features (tempo, energy, danceability, etc.) that have
#      been pre-processed and stored in `transformed_matrix`.
#
#   2. Collaborative Filtering:
#      Recommends songs that are *listened to by similar users*.
#      It looks at user-song interaction patterns stored in `interaction_matrix`
#      (e.g. "users who liked this song also liked these songs").
#
# The final recommendation is a weighted blend of both approaches.
# =============================================================================

class HybridRecommenderSystem:
    
    def __init__(self, song_name: str, 
                 artist_name: str,  
                 number_of_recommendations: int, 
                 weight_content_based: float, 
                 weight_collaborative: float, 
                 songs_data, transformed_matrix, 
                 interaction_matrix, track_ids):
        """
        Initialises the recommender with all the data and parameters it needs.

        Parameters:
        -----------
        song_name               : The name of the input/seed song to base recommendations on.
        artist_name             : The artist of the seed song (needed to disambiguate
                                  songs with the same title by different artists).
        number_of_recommendations: How many song recommendations to return.
        weight_content_based    : How much weight to give content-based similarity (0.0 to 1.0).
        weight_collaborative    : How much weight to give collaborative filtering similarity.
                                  Ideally weight_content_based + weight_collaborative = 1.0
        songs_data              : A DataFrame containing song metadata (name, artist, track_id, etc.)
        transformed_matrix      : A pre-computed feature matrix where each row is a song
                                  represented as a vector of audio features (used for content-based).
        interaction_matrix      : A pre-computed user-song matrix where each row is a song
                                  represented by its user interaction patterns (used for collaborative).
        track_ids               : A NumPy array of track IDs aligned with the rows of interaction_matrix.
        """
        
        self.number_of_recommendations = number_of_recommendations
        # Standardise to lowercase so lookups are case-insensitive
        self.song_name = song_name.lower()
        self.artist_name = artist_name.lower()
        self.weight_content_based = weight_content_based
        self.weight_collaborative = weight_collaborative
        self.songs_data = songs_data
        self.transformed_matrix = transformed_matrix
        self.interaction_matrix = interaction_matrix
        self.track_ids = track_ids
        
        
    def calculate_content_based_similarities(self, song_name, artist_name, songs_data, transformed_matrix):
        """
        Computes how similar every song in the catalogue is to the input song,
        based purely on audio/musical features (content-based approach).

        Steps:
        1. Find the row in songs_data that matches the input song + artist.
        2. Use the row index to pull that song's feature vector from transformed_matrix.
        3. Compute cosine similarity between that vector and ALL other song vectors.
           Cosine similarity returns a value between 0 (completely different) and 1 (identical).

        Returns: A 1 x N array of similarity scores (one score per song in the catalogue).
        """

        # Step 1: Filter the DataFrame to find the exact row for this song + artist combo.
        # Both name and artist must match to avoid ambiguity (e.g. two songs called "Hero").
        song_row = songs_data.loc[(songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)]
        
        # Step 2: Get the integer index (row position) of this song in the DataFrame.
        # This index aligns with the row position in transformed_matrix.
        song_index = song_row.index[0]
        
        # Step 3: Extract the song's feature vector from the transformed matrix.
        # reshape(1, -1) converts it from a flat 1D array to a 2D row vector
        # as required by cosine_similarity (which expects 2D input).
        input_vector = transformed_matrix[song_index].reshape(1, -1)
        
        # Step 4: Compute cosine similarity between this song's vector and every
        # song's vector in the full transformed_matrix. Returns a 1 x N matrix.
        content_similarity_scores = cosine_similarity(input_vector, transformed_matrix)
        
        return content_similarity_scores
        
    
    def calculate_collaborative_filtering_similarities(self, song_name, artist_name, track_ids, songs_data, interaction_matrix):
        """
        Computes how similar every song in the catalogue is to the input song,
        based on user listening behaviour (collaborative filtering approach).

        The interaction_matrix captures patterns like "users who played Song A
        also played Song B" — so songs with similar listener profiles score highly.

        Steps:
        1. Find the track_id of the input song from songs_data.
        2. Locate which row in interaction_matrix corresponds to this track_id.
        3. Compute cosine similarity between that row and all other rows.

        Returns: A 1 x N array of similarity scores.
        """

        # Step 1: Look up the row in songs_data for this song + artist.
        song_row = songs_data.loc[(songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)]
        
        # Step 2: Extract the track_id (a unique identifier like a Spotify track ID).
        # .values.item() converts it from a NumPy array to a plain Python scalar.
        input_track_id = song_row['track_id'].values.item()
        
        # Step 3: Find the index (row position) of this track_id in the track_ids array.
        # np.where returns a tuple of arrays; [0].item() extracts the single integer index.
        # This is needed because interaction_matrix rows are indexed by position,
        # not by track_id directly.
        ind = np.where(track_ids == input_track_id)[0].item()
        
        # Step 4: Pull out the song's row from the interaction matrix.
        # This vector represents how users have interacted with this song
        # (e.g. play counts, saves, or binary listened/not-listened flags).
        input_array = interaction_matrix[ind]
        
        # Step 5: Compute cosine similarity between this song's interaction vector
        # and every other song's interaction vector in the full interaction_matrix.
        collaborative_similarity_scores = cosine_similarity(input_array, interaction_matrix)
        
        return collaborative_similarity_scores
    
    
    def normalize_similarities(self, similarity_scores):
        """
        Scales all similarity scores to a 0–1 range using Min-Max normalisation.

        Why this is necessary:
        The content-based and collaborative scores are computed independently
        and may have very different value ranges or distributions.
        Before combining them with weights, we need them on the same scale —
        otherwise one method could dominate simply because its raw values are larger.

        Formula: normalised = (value - min) / (max - min)
        - The minimum value becomes 0.0
        - The maximum value becomes 1.0
        - Everything else scales proportionally in between.
        """

        minimum = np.min(similarity_scores)
        maximum = np.max(similarity_scores)
        normalized_scores = (similarity_scores - minimum) / (maximum - minimum)
        return normalized_scores
    
    
    def weighted_combination(self, content_based_scores, collaborative_filtering_scores):
        """
        Blends the two normalised similarity scores into a single hybrid score.

        Each score is multiplied by its assigned weight and then summed.
        For example, with weight_content_based=0.3 and weight_collaborative=0.7:
            hybrid_score = (0.3 × content_score) + (0.7 × collaborative_score)

        This means collaborative patterns (what users listen to) matter more
        than audio features in this configuration. The weights let you tune
        the balance based on what works best for your use case.
        """

        weighted_scores = (self.weight_content_based * content_based_scores) + \
                          (self.weight_collaborative * collaborative_filtering_scores)
        return weighted_scores
    
    
    def give_recommendations(self):
        """
        Orchestrates the full recommendation pipeline and returns the top K songs.

        Full pipeline:
        1.  Compute content-based similarity scores for all songs.
        2.  Compute collaborative filtering similarity scores for all songs.
        3.  Normalise both sets of scores to 0–1 range.
        4.  Combine them into a single weighted hybrid score per song.
        5.  Rank all songs by their hybrid score (highest = most recommended).
        6.  Pick the top K+1 songs (the +1 accounts for the input song itself,
            which will score 1.0 and needs to be excluded from results).
        7.  Map the selected indices back to track_ids.
        8.  Look up the full song details from songs_data.
        9.  Return a clean DataFrame of the top K recommendations.
        """

        # ── Step 1: Content-based similarities ──────────────────────────────
        # Returns a 1 x N array: similarity of the input song to every song
        # based on audio features (tempo, energy, danceability, etc.)
        content_based_similarities = self.calculate_content_based_similarities(
            song_name=self.song_name, 
            artist_name=self.artist_name, 
            songs_data=self.songs_data, 
            transformed_matrix=self.transformed_matrix
        )
        
        # ── Step 2: Collaborative filtering similarities ─────────────────────
        # Returns a 1 x N array: similarity of the input song to every song
        # based on shared listener behaviour patterns.
        collaborative_filtering_similarities = self.calculate_collaborative_filtering_similarities(
            song_name=self.song_name, 
            artist_name=self.artist_name, 
            track_ids=self.track_ids, 
            songs_data=self.songs_data, 
            interaction_matrix=self.interaction_matrix
        )
    
        # ── Step 3: Normalise both score arrays to 0–1 ──────────────────────
        # Ensures both methods are on the same scale before combining them.
        normalized_content_based_similarities = self.normalize_similarities(content_based_similarities)
        normalized_collaborative_filtering_similarities = self.normalize_similarities(collaborative_filtering_similarities)
        
        # ── Step 4: Combine into one hybrid score per song ───────────────────
        # Applies the user-defined weights to blend the two similarity signals.
        weighted_scores = self.weighted_combination(
            content_based_scores=normalized_content_based_similarities, 
            collaborative_filtering_scores=normalized_collaborative_filtering_similarities
        )
        
        # ── Step 5: Rank songs by hybrid score ───────────────────────────────
        # .ravel() flattens the 2D array to 1D so argsort works correctly.
        # np.argsort returns indices that would sort the array in ascending order,
        # so we take the LAST (number_of_recommendations + 1) indices.
        # [::-1] reverses to get descending order (highest scores first).
        # We fetch K+1 because the input song itself will be #1 and must be dropped.
        recommendation_indices = np.argsort(weighted_scores.ravel())[-self.number_of_recommendations - 1:][::-1]
        
        # ── Step 6: Map indices back to track_ids ────────────────────────────
        # The indices from argsort correspond to row positions in track_ids,
        # so we use them to look up the actual track_id values.
        recommendation_track_ids = self.track_ids[recommendation_indices]
       
        # ── Step 7: Get the top scores for reference ─────────────────────────
        # Extract the actual hybrid score values for the top K+1 songs.
        # These are used later to sort the output DataFrame correctly.
        top_scores = np.sort(weighted_scores.ravel())[-self.number_of_recommendations - 1:][::-1]
        
        # ── Step 8: Build the output DataFrame ───────────────────────────────
        # Create a temporary scores DataFrame to join scores with track_ids.
        scores_df = pd.DataFrame({
            "track_id": recommendation_track_ids.tolist(),
            "score": top_scores
        })
        
        # Filter songs_data to only the recommended track_ids,
        # merge in the scores, sort by score descending,
        # drop internal columns (track_id and score) that the end user doesn't need,
        # and reset the index to 0, 1, 2... for a clean output.
        # Note: this also implicitly drops the input song itself if it appears
        # in songs_data with its track_id (it merges by track_id match).
        top_k_songs = (
            self.songs_data
            .loc[self.songs_data["track_id"].isin(recommendation_track_ids)]  # keep only recommended songs
            .merge(scores_df, on="track_id")                                  # attach scores
            .sort_values(by="score", ascending=False)                         # highest score first
            .drop(columns=["track_id", "score"])                              # remove internal columns
            .reset_index(drop=True)                                           # clean 0-based index
        )
        
        return top_k_songs
    
    
# =============================================================================
# Entry point — runs when the script is executed directly
# =============================================================================
if __name__ == "__main__":

    # ── Load pre-computed data files ─────────────────────────────────────────

    # The content-based feature matrix: each row = one song as a feature vector
    # (audio attributes like tempo, energy, valence, etc. after transformation/scaling).
    # Stored as a sparse matrix (.npz) to save memory.
    transformed_data = load_npz("data/transformed_hybrid_data.npz")
    
    # The collaborative filtering interaction matrix: each row = one song,
    # each column = one user, values represent interaction strength
    # (e.g. play count or binary played/not-played).
    # Also sparse since most users haven't listened to most songs.
    interaction_matrix = load_npz("data/interaction_matrix.npz")
    
    # A NumPy array of track_ids aligned row-by-row with the interaction_matrix.
    # Needed to translate matrix row indices back to recognisable track identifiers.
    track_ids = np.load("data/track_ids.npy", allow_pickle=True)
    
    # The songs metadata DataFrame — only loading the columns we actually need
    # to keep memory usage lean.
    songs_data = pd.read_csv(
        "data/collab_filtered_data.csv",
        usecols=["track_id", "name", "artist", "spotify_preview_url"]
    )
    
    # ── Instantiate the recommender ──────────────────────────────────────────
    # Here we're asking: "Give me 10 songs similar to Love Story by Taylor Swift,
    # weighting collaborative filtering 70% and content-based 30%."
    # The higher collaborative weight means user listening behaviour matters more
    # than pure audio similarity in this configuration.
    hybrid_recommender = HybridRecommenderSystem(
        song_name="Love Story",
        artist_name="Taylor Swift", 
        number_of_recommendations=10, 
        weight_content_based=0.3,       # 30% — audio feature similarity
        weight_collaborative=0.7,       # 70% — user behaviour similarity
        songs_data=songs_data, 
        transformed_matrix=transformed_data, 
        interaction_matrix=interaction_matrix, 
        track_ids=track_ids
    )
    
    # ── Run the recommender and print results ────────────────────────────────
    # Returns a DataFrame with columns: name, artist, spotify_preview_url
    # sorted by hybrid similarity score descending.
    recommendations = hybrid_recommender.give_recommendations()
    
    print(recommendations)