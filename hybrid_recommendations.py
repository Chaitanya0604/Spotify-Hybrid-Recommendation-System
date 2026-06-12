import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


class HybridRecommenderSystem:
    
    def __init__(self,  
                 number_of_recommendations: int, 
                 weight_content_based: float):
        """
        Initialise the Hybrid Recommender System.
        
        Args:
            number_of_recommendations: Number of songs to recommend (k)
            weight_content_based: Weight for content-based scores (0.0 - 1.0)
                                  Collaborative weight is automatically set as 1 - content_weight
        """
        # Store how many songs to recommend
        self.number_of_recommendations = number_of_recommendations
        
        # Store the content-based weight (set by user via diversity slider)
        # e.g. diversity=3 → content_weight=0.7 (lean more towards audio features)
        # e.g. diversity=8 → content_weight=0.2 (lean more towards listening patterns)
        self.weight_content_based = weight_content_based
        
        # Collaborative weight is always the complement of content-based weight
        # Both weights always sum to 1.0, ensuring a balanced combination
        # e.g. content=0.3 → collaborative=0.7
        self.weight_collaborative = 1 - weight_content_based
        
        
    def __calculate_content_based_similarities(self, song_name, artist_name, songs_data, transformed_matrix):
        """
        Compute cosine similarity between the input song and ALL songs
        using the content-based feature matrix (audio features + tags).
        
        How it works:
            - Finds the input song's row in the feature matrix
            - Computes cosine similarity: measures the angle between two feature vectors
            - Score of 1.0 = identical audio profile, 0.0 = completely different
        
        Returns:
            content_similarity_scores: 1 × N array of similarity scores (one per song)
        """
        # ── Step 1: Locate the input song in the dataset ─────────────────────
        # Filter songs_data by both name AND artist to avoid matching
        # songs with the same title by different artists
        song_row = songs_data.loc[(songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)]
        
        # ── Step 2: Get its positional index ─────────────────────────────────
        # This index directly maps to the same row in the transformed_matrix
        # (both dataset and matrix share the same row order — sorted by track_id)
        song_index = song_row.index[0]
        
        # ── Step 3: Extract the song's feature vector ─────────────────────────
        # transformed_matrix[song_index] gives a 1D array of the song's features
        # reshape(1, -1) converts it to (1, n_features) — required format for cosine_similarity
        input_vector = transformed_matrix[song_index].reshape(1, -1)
        
        # ── Step 4: Compute cosine similarity against every song ──────────────
        # Compares input_vector (1 × n_features) against transformed_matrix (N × n_features)
        # Result shape: (1, N) — one similarity score per song in the dataset
        content_similarity_scores = cosine_similarity(input_vector, transformed_matrix)
        
        return content_similarity_scores
        
    
    def __calculate_collaborative_filtering_similarities(self, song_name, artist_name, track_ids, songs_data, interaction_matrix):
        """
        Compute cosine similarity between the input song and ALL songs
        using the user-song interaction matrix (listening history patterns).
        
        How it works:
            - Each row in the interaction matrix represents a song
            - Each column represents a user
            - A value > 0 means that user has listened to that song
            - Cosine similarity finds songs listened to by similar users
            - Songs co-listened frequently by the same users score higher
        
        Returns:
            collaborative_similarity_scores: 1 × N array of similarity scores (one per song)
        """
        # ── Step 1: Locate the input song in the dataset ─────────────────────
        # Filter by both name and artist for an exact match
        song_row = songs_data.loc[(songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)]
        
        # ── Step 2: Get the track_id of the input song ────────────────────────
        # track_id is the unique Spotify identifier for each song
        # .values.item() extracts the scalar value from the array
        input_track_id = song_row['track_id'].values.item()
        
        # ── Step 3: Find the track_id's position in the track_ids array ───────
        # track_ids is a sorted numpy array of all track IDs in the interaction matrix
        # np.where returns the index where input_track_id appears
        # This index maps track_id → row position in the interaction_matrix
        # .item() extracts the scalar integer from the returned array
        ind = np.where(track_ids == input_track_id)[0].item()
        
        # ── Step 4: Extract the song's interaction vector ─────────────────────
        # interaction_matrix[ind] gives a sparse row vector of shape (1, n_users)
        # Each non-zero value represents a user who has listened to this song
        input_array = interaction_matrix[ind]
        
        # ── Step 5: Compute cosine similarity against every song ──────────────
        # Compares input_array (1 × n_users) against interaction_matrix (N × n_users)
        # Songs listened to by the same users will have higher similarity scores
        # Result shape: (1, N) — one similarity score per song in the dataset
        collaborative_similarity_scores = cosine_similarity(input_array, interaction_matrix)
        
        return collaborative_similarity_scores
    
    
    def __normalize_similarities(self, similarity_scores):
        """
        Apply Min-Max Normalization to scale all scores to the range [0, 1].
        
        Why this is critical:
            - Content-based scores are dense → values tend to be large (close to 1)
            - Collaborative scores are sparse → values tend to be very small (close to 0)
            - Adding them directly means content-based always dominates,
              even if its weight is set to just 20%
            - Normalization levels the playing field so weights work as intended
        
        Formula: normalized = (x - min) / (max - min)
            - The minimum score becomes 0.0
            - The maximum score becomes 1.0
            - All other scores scale proportionally in between
        """
        # Find the lowest similarity score in the array
        minimum = np.min(similarity_scores)
        
        # Find the highest similarity score in the array
        maximum = np.max(similarity_scores)
        
        # Apply Min-Max formula — shifts and scales all values to [0, 1]
        # After this, both content and collaborative arrays live on the same scale
        normalized_scores = (similarity_scores - minimum) / (maximum - minimum)
        
        return normalized_scores
    
    
    def __weighted_combination(self, content_based_scores, collaborative_filtering_scores):
        """
        Combine normalized content and collaborative scores into a single hybrid score.
        
        Formula: final_score = (weight_content × content_norm) + (weight_collab × collab_norm)
        
        How the weights work:
            - Both weights always sum to 1.0
            - Higher content weight → recommendations lean towards audio similarity
            - Higher collaborative weight → recommendations lean towards listening patterns
            - The user controls this balance via the diversity slider in the app:
                Slider=1 → content_weight=0.9  (mostly audio-based)
                Slider=5 → content_weight=0.5  (equal blend)
                Slider=10 → content_weight=0.0 (fully listening-pattern-based)
        """
        # Multiply each score array by its respective weight and sum element-wise
        # Both arrays are shape (1, N) so this operation is straightforward
        # The result is also (1, N) — one final hybrid score per song
        weighted_scores = (self.weight_content_based * content_based_scores) + (self.weight_collaborative * collaborative_filtering_scores)
        
        return weighted_scores
    
    
    def give_recommendations(self, song_name, artist_name, songs_data, track_ids, transformed_matrix, interaction_matrix):
        """
        Main method — orchestrates the full hybrid recommendation pipeline.
        
        Full Pipeline:
            1. Calculate content-based similarity scores  (audio features)
            2. Calculate collaborative similarity scores  (listening history)
            3. Normalize both arrays to [0, 1]           (level the scale)
            4. Combine using weighted addition            (user-controlled blend)
            5. Rank all songs by hybrid score             (best matches first)
            6. Retrieve and return top-k songs            (final output)
            
        Returns:
            top_k_songs (DataFrame): Top k recommended songs sorted by hybrid score,
                                     including the input song itself at position 0
        """
        
        # ── Step 1: Content-Based Similarities ──────────────────────────────────
        # Uses the transformed feature matrix (audio features + tags)
        # Returns shape (1, N) — cosine similarity of input song vs all 30K songs
        content_based_similarities = self.__calculate_content_based_similarities(
            song_name=song_name, 
            artist_name=artist_name, 
            songs_data=songs_data, 
            transformed_matrix=transformed_matrix
        )
        
        # ── Step 2: Collaborative Filtering Similarities ─────────────────────────
        # Uses the user-song interaction matrix (listening history)
        # Returns shape (1, N) — cosine similarity of input song vs all 30K songs
        # based on which users listened to which songs
        collaborative_filtering_similarities = self.__calculate_collaborative_filtering_similarities(
            song_name=song_name, 
            artist_name=artist_name, 
            track_ids=track_ids, 
            songs_data=songs_data, 
            interaction_matrix=interaction_matrix
        )
    
        # ── Step 3: Normalize Both Score Arrays ──────────────────────────────────
        # Scale content scores from their raw range → [0, 1]
        # Without this, content scores (dense, large) would dominate the final score
        normalized_content_based_similarities = self.__normalize_similarities(content_based_similarities)
        
        # Scale collaborative scores from their raw range → [0, 1]
        # Without this, collaborative scores (sparse, tiny) would be drowned out
        normalized_collaborative_filtering_similarities = self.__normalize_similarities(collaborative_filtering_similarities)
        
        # ── Step 4: Weighted Combination ─────────────────────────────────────────
        # Blend both normalized arrays using the user-defined weights
        # final_score = (content_weight × content_norm) + (collab_weight × collab_norm)
        # Result shape: (1, N) — one hybrid score per song
        weighted_scores = self.__weighted_combination(
            content_based_scores=normalized_content_based_similarities, 
            collaborative_filtering_scores=normalized_collaborative_filtering_similarities
        )
        
        # ── Step 5: Rank All Songs by Hybrid Score ────────────────────────────────
        
        # ravel() flattens (1, N) → (N,) for easier sorting
        # argsort() returns indices that would sort the array in ascending order
        # [-k-1:] slices the last (k+1) indices → the highest scoring songs
        # +1 because index 0 will be the input song itself (similarity = 1.0 with itself)
        # [::-1] reverses to get descending order (highest score first)
        recommendation_indices = np.argsort(weighted_scores.ravel())[-self.number_of_recommendations - 1:][::-1]
        
        # ── Step 6: Retrieve Top-K Songs ─────────────────────────────────────────
        
        # Map the matrix row indices back to their corresponding track_ids
        recommendation_track_ids = track_ids[recommendation_indices]
       
        # Extract the actual hybrid scores for the top-k songs (in the same order)
        top_scores = np.sort(weighted_scores.ravel())[-self.number_of_recommendations - 1:][::-1]
        
        # Build a small DataFrame linking each track_id to its hybrid score
        # This will be used to merge scores back into the songs metadata
        scores_df = pd.DataFrame({
            "track_id": recommendation_track_ids.tolist(),
            "score": top_scores
        })
        
        # Filter songs_data to only the recommended track_ids
        # Merge to attach hybrid scores to each song's metadata
        # Sort by score descending so the best recommendation appears first
        # Drop track_id and score columns — these are internal, not needed in output
        # Reset index to give clean 0-based indexing (index 0 = input song)
        top_k_songs = (
            songs_data
            .loc[songs_data["track_id"].isin(recommendation_track_ids)]  # keep only top-k songs
            .merge(scores_df, on="track_id")                              # attach hybrid scores
            .sort_values(by="score", ascending=False)                     # best match first
            .drop(columns=["track_id", "score"])                          # clean up helper columns
            .reset_index(drop=True)                                       # clean 0-based index
        )
        
        return top_k_songs