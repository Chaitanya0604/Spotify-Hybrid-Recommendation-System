import streamlit as st
from content_based_filtering import content_recommendation
from scipy.sparse import load_npz
import pandas as pd
from numpy import load
from hybrid_recommendations import HybridRecommenderSystem

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Songbird · Music Recommender",
    page_icon="🎼",
    layout="centered"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');

/* ── Reset & base ── */
.stApp {
    background-color: #FAFAF8;
    font-family: 'Inter', sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 720px; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 56px 0 40px;
}
.hero-eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #8BAF9A;
    margin-bottom: 16px;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.6rem;
    font-weight: 700;
    color: #1C1C1E;
    line-height: 1.1;
    margin: 0 0 8px;
}
.hero-title em {
    font-style: italic;
    color: #E8A598;
}
.hero-sub {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    font-weight: 300;
    color: #6B6B6B;
    margin-top: 12px;
    line-height: 1.6;
}
.hero-rule {
    width: 48px;
    height: 2px;
    background: linear-gradient(90deg, #E8A598, #C4B5D4);
    margin: 28px auto 0;
    border-radius: 2px;
}

/* ── Section label ── */
.section-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #AAAAAA;
    margin: 32px 0 10px;
}

/* ── Text inputs ── */
.stTextInput > div > div > input {
    background-color: #FFFFFF;
    border: 1.5px solid #E8E8E4;
    border-radius: 12px;
    padding: 12px 16px;
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    color: #1C1C1E !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    transition: border-color 0.2s ease;
}
.stTextInput > div > div > input:focus {
    border-color: #E8A598;
    box-shadow: 0 0 0 3px rgba(232,165,152,0.15);
}
.stTextInput > label {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    color: #555;
}

/* ── Selectbox — fix invisible text ── */
.stSelectbox > div > div {
    background-color: #FFFFFF !important;
    border: 1.5px solid #E8E8E4 !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif;
}
.stSelectbox > label {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    color: #555;
}
/* The actual selected value text */
.stSelectbox [data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #1C1C1E !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    border: 1.5px solid #E8E8E4 !important;
    border-radius: 12px !important;
}
.stSelectbox [data-baseweb="select"] span,
.stSelectbox [data-baseweb="select"] div {
    color: #1C1C1E !important;
}
/* Dropdown list items */
[data-baseweb="popover"] [role="option"] {
    font-family: 'Inter', sans-serif !important;
    color: #1C1C1E !important;
    background-color: #FFFFFF !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] {
    background-color: #FDF5F3 !important;
    color: #1C1C1E !important;
}

/* ── Slider ── */
div[data-testid="stSlider"] label {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    color: #555;
}
/* Thumb (the draggable handle) — purple dot */
div[data-testid="stSlider"] div[role="slider"] {
    background-color: #C4B5D4 !important;
    border-color: #B09EC8 !important;
    box-shadow: 0 0 0 4px rgba(196,181,212,0.28) !important;
}
/* Value bubble shown above the thumb while dragging */
div[data-testid="stSlider"] div[role="slider"] div {
    background-color: #1C1C1E !important;
    color: #FFFFFF !important;
}
/* Filled portion of the track — soft pink */
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div[style] {
    background: #F2C4BA !important;
}
/* Unfilled portion of the track stays a soft neutral grey */
div[data-testid="stSlider"] div[data-baseweb="slider"] > div {
    background-color: #EEEDE9 !important;
}
/* Min / max tick labels */
div[data-testid="stSlider"] [data-testid="stTickBarMin"],
div[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    font-family: 'Inter', sans-serif;
    color: #AAAAAA;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #E8A598 0%, #C4B5D4 100%);
    color: #1C1C1E;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.3px;
    border: none;
    border-radius: 50px;
    padding: 12px 40px;
    margin-top: 8px;
    box-shadow: 0 4px 16px rgba(232,165,152,0.35);
    transition: all 0.2s ease;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(232,165,152,0.45);
}

/* ── Divider ── */
.stMarkdown hr {
    border: none;
    border-top: 1px solid #EEEDE9;
    margin: 28px 0;
}

/* ── Song cards ── */
.song-card {
    background: #FFFFFF;
    border-radius: 20px;
    padding: 0;
    margin-bottom: 14px;
    box-shadow: 0 2px 16px rgba(232,165,152,0.13);
    border: 2px solid #F2C4BA;
    overflow: hidden;
    display: flex;
    align-items: stretch;
    transition: box-shadow 0.25s ease, transform 0.25s ease, border-color 0.25s ease;
}
.song-card:hover {
    box-shadow: 0 8px 28px rgba(232,165,152,0.32);
    transform: translateY(-2px) scale(1.012);
    border-color: #E8A598;
}
.song-card.featured {
    background: linear-gradient(135deg, #FDEAE6 0%, #F5EBF8 100%);
    border: none;
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(196,181,212,0.25);
}
.card-number-band {
    width: 56px;
    min-height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    background: rgba(232,165,152,0.12);
    border-right: 1px solid #F0D5D0;
}
.song-card.featured .card-number-band {
    background: linear-gradient(160deg, #E8A598, #C4B5D4);
    border-right: none;
}
.card-number {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    font-weight: 700;
    font-style: italic;
    color: #BBBBBB;
}
.song-card.featured .card-number {
    color: #FFFFFF;
}
.card-body {
    padding: 18px 20px;
    flex: 1;
}
.card-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #E8A598;
    margin-bottom: 5px;
}
.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #1C1C1E;
    margin: 0 0 3px;
    line-height: 1.3;
}
.card-artist {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    color: #999;
    margin: 0;
}
.card-note {
    font-size: 1.1rem;
    opacity: 0.18;
    margin-left: auto;
    padding: 18px 20px 18px 0;
    display: flex;
    align-items: center;
    flex-shrink: 0;
    font-size: 1.5rem;
}

/* ── Results header ── */
.results-for {
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
    color: #888;
    margin-bottom: 20px;
}
.results-for strong {
    color: #1C1C1E;
    font-weight: 600;
}

/* ── Diversity bar chart ── */
.barchart-widget {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 24px 24px 18px;
    border: 1px solid #F0EFE9;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    margin-top: 12px;
}
.barchart-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #BBBBBB;
    text-align: center;
    margin-bottom: 18px;
}
.barchart-bars {
    display: flex;
    align-items: flex-end;
    justify-content: center;
    gap: 56px;
    height: 130px;
}
.bar-col {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 64px;
}
.bar-value {
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-weight: 700;
    font-size: 1.15rem;
    color: #1C1C1E;
    margin-bottom: 8px;
}
.bar-rect {
    width: 100%;
    border-radius: 10px 10px 3px 3px;
    transition: height 0.35s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
.bar-rect.personal { background: linear-gradient(180deg, #f0bdb1 0%, #E8A598 100%); }
.bar-rect.diverse  { background: linear-gradient(180deg, #d9cce7 0%, #C4B5D4 100%); }
.barchart-baseline {
    width: 220px;
    height: 2px;
    background: #EEEDE9;
    margin: 0 auto;
}
.barchart-labels {
    display: flex;
    justify-content: center;
    gap: 56px;
    margin-top: 10px;
}
.bar-label-col {
    width: 64px;
    text-align: center;
}
.bar-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.bar-label.personal { color: #E8A598; }
.bar-label.diverse  { color: #C4B5D4; }
.bar-desc {
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    color: #BBBBBB;
    margin-top: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────
songs_data              = pd.read_csv("data/cleaned_data.csv")
transformed_data        = load_npz("data/transformed_data.npz")
track_ids               = load("data/track_ids.npy", allow_pickle=True)
filtered_data           = pd.read_csv("data/collab_filtered_data.csv")
interaction_matrix      = load_npz("data/interaction_matrix.npz")
transformed_hybrid_data = load_npz("data/transformed_hybrid_data.npz")

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Music Discovery</div>
    <div class="hero-title">Find your next<br><em>favourite song</em></div>
    <div class="hero-sub">Tell us what you're listening to.<br>We'll find what comes next.</div>
    <div class="hero-rule"></div>
</div>
""", unsafe_allow_html=True)

# ── Inputs ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Search</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    song_name = st.text_input("Song name", placeholder="e.g. Love Story")
with col2:
    artist_name = st.text_input("Artist name", placeholder="e.g. Taylor Swift")

song_name   = song_name.lower()
artist_name = artist_name.lower()

k = st.selectbox("How many recommendations?", [5, 10, 15, 20], index=1)

# ── Filtering type + diversity ─────────────────────────────────────────────────
if ((filtered_data["name"] == song_name) & (filtered_data["artist"] == artist_name)).any():
    filtering_type = "Hybrid Recommender System"
    st.markdown("---")
    st.markdown('<div class="section-label">Tune your taste</div>', unsafe_allow_html=True)
    diversity = st.slider(
        "Personalized  ◂────────▸  Diverse",
        min_value=1, max_value=9, value=5, step=1
    )
    content_based_weight = 1 - (diversity / 10)
    personal_pct = (10 - diversity) * 10   # e.g. diversity=5 → personal=50
    diverse_pct  = diversity * 10
    max_bar_height = 110  # px, matches .barchart-bars height minus label space
    min_bar_height = 16   # px, keeps a sliver visible even near 0%
    personal_h = max(min_bar_height, round(max_bar_height * personal_pct / 100))
    diverse_h  = max(min_bar_height, round(max_bar_height * diverse_pct  / 100))
    st.markdown(f"""
    <div class="barchart-widget">
        <div class="barchart-title">Recommendation Mix</div>
        <div class="barchart-bars">
            <div class="bar-col">
                <div class="bar-value">{personal_pct}%</div>
                <div class="bar-rect personal" style="height:{personal_h}px;"></div>
            </div>
            <div class="bar-col">
                <div class="bar-value">{diverse_pct}%</div>
                <div class="bar-rect diverse" style="height:{diverse_h}px;"></div>
            </div>
        </div>
        <div class="barchart-baseline"></div>
        <div class="barchart-labels">
            <div class="bar-label-col">
                <div class="bar-label personal">Personalized</div>
                <div class="bar-desc">Similar to your pick</div>
            </div>
            <div class="bar-label-col">
                <div class="bar-label diverse">Diverse</div>
                <div class="bar-desc">New discoveries</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    filtering_type = "Content-Based Filtering"

st.markdown("---")

# ── Card renderer ──────────────────────────────────────────────────────────────
def render_card(display_index, rec, featured=False):
    title  = rec["name"].title()
    artist = rec["artist"].title()
    url    = rec["spotify_preview_url"]
    card_class  = "song-card featured" if featured else "song-card"
    label_text  = "Now playing" if featured else "Up next"
    num_display = "♪" if featured else f"{display_index}"
    st.markdown(f"""
    <div class="{card_class}">
        <div class="card-number-band">
            <span class="card-number">{num_display}</span>
        </div>
        <div class="card-body">
            <div class="card-label">{label_text}</div>
            <div class="card-title">{title}</div>
            <div class="card-artist">{artist}</div>
        </div>
        <div class="card-note">♫</div>
    </div>
    """, unsafe_allow_html=True)
    st.audio(url)

# ── Recommendations ────────────────────────────────────────────────────────────
if filtering_type == "Content-Based Filtering":
    if st.button("Discover songs  →"):
        if ((songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)).any():
            st.markdown(
                f'<div class="results-for">Showing results for '
                f'<strong>{song_name.title()}</strong> by <strong>{artist_name.title()}</strong></div>',
                unsafe_allow_html=True
            )
            recommendations = content_recommendation(
                song_name=song_name,
                artist_name=artist_name,
                songs_data=songs_data,
                transformed_data=transformed_data,
                k=k
            )
            for display_i, (ind, rec) in enumerate(recommendations.iterrows()):
                render_card(display_i, rec, featured=(display_i == 0))
        else:
            st.warning(f"We couldn't find **{song_name.title()}** in our library. Try checking the spelling or a different song.")

elif filtering_type == "Hybrid Recommender System":
    if st.button("Discover songs  →"):
        st.markdown(
            f'<div class="results-for">Showing results for '
            f'<strong>{song_name.title()}</strong> by <strong>{artist_name.title()}</strong></div>',
            unsafe_allow_html=True
        )
        recommender = HybridRecommenderSystem(
            number_of_recommendations=k,
            weight_content_based=content_based_weight
        )
        recommendations = recommender.give_recommendations(
            song_name=song_name,
            artist_name=artist_name,
            songs_data=filtered_data,
            transformed_matrix=transformed_hybrid_data,
            track_ids=track_ids,
            interaction_matrix=interaction_matrix
        )
        for display_i, (ind, rec) in enumerate(recommendations.iterrows()):
            render_card(display_i, rec, featured=(display_i == 0))