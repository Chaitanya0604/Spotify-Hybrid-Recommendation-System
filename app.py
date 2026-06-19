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
    background:
        radial-gradient(ellipse 900px 500px at 15% 0%, rgba(232,165,152,0.07), transparent 60%),
        radial-gradient(ellipse 900px 600px at 100% 30%, rgba(196,181,212,0.08), transparent 60%),
        radial-gradient(ellipse 800px 500px at 10% 100%, rgba(139,175,154,0.05), transparent 60%),
        #FAF7F2;
    font-family: 'Inter', sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 920px; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 56px 0 40px;
}
.hero-eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #8BAF9A;
    margin-bottom: 18px;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 4.2rem;
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
    font-size: 1.1rem;
    font-weight: 300;
    color: #1C1C1E;
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
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #9C86B8;
    margin: 32px 0 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::before {
    content: "";
    width: 14px;
    height: 2px;
    background: linear-gradient(90deg, #E8A598, #C4B5D4);
    border-radius: 2px;
}

/* ── Hide the "Press Enter to apply" instruction text under text inputs ── */
[data-testid="InputInstructions"] {
    display: none !important;
}

/* ── Text inputs ──
   Streamlit/baseweb applies its own border to the wrapping div, which was
   winning over the input's own border and showing as a heavy black edge.
   Targeting the wrapper directly (not just the input) and using !important
   ensures the soft warm border actually renders. */
div[data-testid="stTextInput"] > div {
    background: linear-gradient(135deg, #2B2A2E 0%, #3A3340 100%) !important;
    border: 1.5px solid #423D40 !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 10px rgba(28,28,30,0.18) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stTextInput"] > div:focus-within {
    border-color: #E8A598 !important;
    box-shadow: 0 0 0 3px rgba(232,165,152,0.22) !important;
}
.stTextInput > div > div > input {
    background-color: transparent !important;
    border: none !important;
    padding: 14px 18px;
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-weight: 600;
    font-size: 1.2rem;
    color: #F2C4BA !important;
    box-shadow: none !important;
    caret-color: #F2C4BA;
}
.stTextInput > div > div > input::placeholder {
    color: #8A8390 !important;
    font-style: italic;
    opacity: 1;
}
.stTextInput > label,
.stTextInput > label p {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem !important;
    font-weight: 600;
    color: #8BAF9A !important;
    letter-spacing: 0.3px;
}

/* ── Selectbox — fix invisible box, match input styling ──
   The previous rule assumed a fixed DOM depth (stSelectbox > div > div),
   which doesn't reliably land on the actual visible control box across
   Streamlit versions — that's why it looked unstyled. Targeting the
   baseweb control container directly (and its parent, transparently) is
   robust regardless of how many wrapper divs sit in between.
   min-height + flex centering on the value container fixes the number
   getting visually clipped at the bottom of the box. */
div[data-testid="stSelectbox"] [data-baseweb="select"] {
    background-color: transparent !important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] > div:first-child {
    background-color: #FDFCFE !important;
    border: 1.5px solid #E4DAEF !important;
    border-radius: 14px !important;
    box-shadow: 0 1px 4px rgba(28,28,30,0.04) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
    min-height: 48px !important;
    display: flex !important;
    align-items: center !important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] > div:first-child:hover {
    border-color: #9C86B8 !important;
    box-shadow: 0 0 0 4px rgba(156,134,184,0.24) !important;
}
div[data-testid="stSelectbox"]:has([data-baseweb="select"][aria-expanded="true"]) [data-baseweb="select"] > div:first-child {
    border-color: #7D63A0 !important;
    box-shadow: 0 0 0 4px rgba(156,134,184,0.32) !important;
}
.stSelectbox > label,
.stSelectbox > label p {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem !important;
    font-weight: 600;
    color: #8BAF9A !important;
    letter-spacing: 0.3px;
}
.stSelectbox [data-baseweb="select"] > div {
    background-color: transparent !important;
    color: #4A4550 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1.05rem !important;
    border: none !important;
    padding: 4px 4px !important;
    line-height: 1.4 !important;
    overflow: visible !important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] * {
    color: #4A4550 !important;
    -webkit-text-fill-color: #4A4550 !important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] svg {
    fill: #7D63A0 !important;
}
[data-baseweb="popover"] [role="option"] {
    font-family: 'Inter', sans-serif !important;
    color: #1C1C1E !important;
    background-color: #FFFFFF !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] {
    background-color: #F2ECF8 !important;
    color: #1C1C1E !important;
}

/* ── Slider ── */
div[data-testid="stSlider"] {
    padding: 14px 16px 10px !important;
    border: 1px solid rgba(212,175,55,0.35) !important;
    border-radius: 14px !important;
}
div[data-testid="stSlider"] label,
div[data-testid="stSlider"] label p {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem !important;
    font-weight: 600;
    color: #6FA384;
}
div[data-testid="stSlider"] div[role="slider"] {
    background-color: #C4B5D4 !important;
    border-color: #B09EC8 !important;
    box-shadow: 0 0 0 4px rgba(196,181,212,0.28) !important;
}
div[data-testid="stSlider"] div[role="slider"] div {
    background-color: #1C1C1E !important;
    color: #FFFFFF !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div[style] {
    background: #F2C4BA !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] > div {
    background-color: #EEEDE9 !important;
}
div[data-testid="stSlider"] [data-testid="stTickBarMin"],
div[data-testid="stSlider"] [data-testid="stSliderTickBarMin"],
div[data-testid="stSlider"] [data-testid="stTickBar"] > div:first-child {
    font-family: 'Inter', sans-serif !important;
    color: #E8A598 !important;
    font-weight: 600 !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #E8A598 !important;
}
div[data-testid="stSlider"] [data-testid="stTickBarMax"],
div[data-testid="stSlider"] [data-testid="stSliderTickBarMax"],
div[data-testid="stSlider"] [data-testid="stTickBar"] > div:last-child {
    font-family: 'Inter', sans-serif !important;
    color: #B09EC8 !important;
    font-weight: 600 !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #B09EC8 !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #E8A598 0%, #C4B5D4 100%);
    color: #FFFFFF;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    letter-spacing: 0.3px;
    border: none;
    border-radius: 50px;
    padding: 13px 40px;
    margin-top: 8px;
    box-shadow: 0 4px 16px rgba(196,142,170,0.32);
    transition: all 0.2s ease;
    width: 100%;
}
.stButton > button p {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(196,142,170,0.42);
}

/* ── Warning / alert ──
   Streamlit's default warning style was rendering pale yellow text on a
   pale yellow background — essentially unreadable. Restyled to match the
   page's palette with real contrast: warm ochre text on a soft ochre tint. */
div[data-testid="stAlert"] {
    background-color: #FBF1DF !important;
    border: 1.5px solid #EBD8A8 !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
    box-shadow: 0 1px 4px rgba(28,28,30,0.04);
}
div[data-testid="stAlert"] p {
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    color: #7A5A1E !important;
    line-height: 1.5;
}
div[data-testid="stAlert"] svg {
    fill: #C9963B !important;
}

/* ── Divider ── */
.stMarkdown hr {
    border: none;
    border-top: 1px solid #EEEDE9;
    margin: 28px 0;
}

/* ══════════════════════════════════════════════════════════════════════════
   SONG CARDS — vinyl-sleeve motif.
   Each card reads as a record sleeve: a colored spine down the left edge
   (like an LP spine label, ordered by track position), a spinning-disc
   badge instead of a flat number tile, and the preview player presented
   as a labelled "groove" rather than a bolted-on widget.
   ══════════════════════════════════════════════════════════════════════════ */

div[class*="st-key-content_card_"],
div[class*="st-key-hybrid_card_"] {
    background: #FFFFFF;
    border-radius: 16px;
    margin-bottom: 24px;
    box-shadow: 0 2px 14px rgba(28,28,30,0.06);
    border: 1px solid #EFEAE6;
    overflow: hidden;
    transition: box-shadow 0.25s ease, transform 0.25s ease, border-color 0.25s ease;
    padding: 0 !important;
    position: relative;
}
/* The card clips its own contents (overflow: hidden) so the audio widget's
   native rounded corners and shadow never float outside the card's shape —
   the audio wrapper below is rounded to match the card's bottom corners
   instead, so it reads as one continuous object, not two stacked shapes. */
div[class*="st-key-content_card_"]:hover,
div[class*="st-key-hybrid_card_"]:hover {
    box-shadow: 0 10px 26px rgba(28,28,30,0.10);
    transform: translateY(-2px);
}
div[class*="st-key-content_card_"]:has(.spine-0):hover,
div[class*="st-key-hybrid_card_"]:has(.spine-0):hover { border-color: #E8A598; }
div[class*="st-key-content_card_"]:has(.spine-1):hover,
div[class*="st-key-hybrid_card_"]:has(.spine-1):hover { border-color: #C4B5D4; }
div[class*="st-key-content_card_"]:has(.spine-2):hover,
div[class*="st-key-hybrid_card_"]:has(.spine-2):hover { border-color: #8BAF9A; }
div[class*="st-key-content_card_"]:has(.spine-3):hover,
div[class*="st-key-hybrid_card_"]:has(.spine-3):hover { border-color: #D9B88A; }
div[class*="st-key-content_card_"]:has(.spine-featured):hover,
div[class*="st-key-hybrid_card_"]:has(.spine-featured):hover { border-color: #B8AEC0; }

/* Spine — the LP-sleeve label strip, color cycles down the stack so
   position in the list is legible at a glance without relying on the
   (now decorative) disc number alone. Explicit per-index classes are used
   instead of inline CSS custom properties so the cycle is guaranteed to
   render regardless of how the host sanitizes inline style attributes. */
.song-card-top {
    border-radius: 16px 16px 0 0;
    padding: 24px 26px 20px 30px;
    display: flex;
    align-items: center;
    gap: 20px;
    position: relative;
}
.song-card-top::before {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 5px;
}
.song-card-top.spine-0::before { background: #E8A598; }
.song-card-top.spine-1::before { background: #C4B5D4; }
.song-card-top.spine-2::before { background: #8BAF9A; }
.song-card-top.spine-3::before { background: #D9B88A; }
.song-card-top.spine-featured::before { background: #E8A598; }

.song-card-top.spine-0 { background: linear-gradient(180deg, #FBF1EE 0%, #FDF8F7 100%); }
.song-card-top.spine-1 { background: linear-gradient(180deg, #F6F2F8 0%, #FBF9FC 100%); }
.song-card-top.spine-2 { background: linear-gradient(180deg, #F1F6F3 0%, #FAFCFA 100%); }
.song-card-top.spine-3 { background: linear-gradient(180deg, #FAF4EB 0%, #FDFAF6 100%); }

/* Disc badge — concentric rings via layered radial-gradients, no image
   assets. The featured ("now playing") card's disc spins continuously;
   all others sit still, so motion itself marks the active track. */
.card-disc {
    width: 52px;
    height: 52px;
    border-radius: 50%;
    flex-shrink: 0;
    position: relative;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06), 0 2px 6px rgba(0,0,0,0.18);
    display: flex;
    align-items: center;
    justify-content: center;
}
.card-disc::after {
    content: "";
    width: 5px;
    height: 5px;
    border-radius: 50%;
}
.card-disc.disc-0 {
    background: radial-gradient(circle at center, #1C1C1E 0 5px, transparent 5.5px),
        repeating-radial-gradient(circle at center, #2A211A 0 2px, #3D3023 2px 5px);
}
.card-disc.disc-0::after { background: #E8A598; }
.card-disc.disc-1 {
    background: radial-gradient(circle at center, #1C1C1E 0 5px, transparent 5.5px),
        repeating-radial-gradient(circle at center, #241F2A 0 2px, #352F3D 2px 5px);
}
.card-disc.disc-1::after { background: #C4B5D4; }
.card-disc.disc-2 {
    background: radial-gradient(circle at center, #1C1C1E 0 5px, transparent 5.5px),
        repeating-radial-gradient(circle at center, #1B221E 0 2px, #28332C 2px 5px);
}
.card-disc.disc-2::after { background: #8BAF9A; }
.card-disc.disc-3 {
    background: radial-gradient(circle at center, #1C1C1E 0 5px, transparent 5.5px),
        repeating-radial-gradient(circle at center, #271F17 0 2px, #3A2F22 2px 5px);
}
.card-disc.disc-3::after { background: #D9B88A; }
.card-disc.disc-featured {
    background: radial-gradient(circle at center, #1C1C1E 0 5px, transparent 5.5px),
        repeating-radial-gradient(circle at center, #2A2A2C 0 2px, #3D3D40 2px 5px);
}
.card-disc.disc-featured::after { background: #E8A598; }
.card-disc.spin {
    animation: disc-spin 5s linear infinite;
}
@keyframes disc-spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
@media (prefers-reduced-motion: reduce) {
    .card-disc.spin { animation: none; }
}

.card-body {
    flex: 1;
    min-width: 0;
}
.card-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    margin-bottom: 5px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.card-label.label-0 { color: #E8A598; }
.card-label.label-1 { color: #C4B5D4; }
.card-label.label-2 { color: #8BAF9A; }
.card-label.label-3 { color: #D9B88A; }
.card-label.label-featured { color: #E8A598; }
.card-label .live-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #E8A598;
    box-shadow: 0 0 0 0 rgba(232,165,152,0.6);
    animation: live-pulse 1.8s ease-out infinite;
}
@keyframes live-pulse {
    0%   { box-shadow: 0 0 0 0 rgba(232,165,152,0.55); }
    70%  { box-shadow: 0 0 0 6px rgba(232,165,152,0); }
    100% { box-shadow: 0 0 0 0 rgba(232,165,152,0); }
}
.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #1C1C1E;
    margin: 0 0 4px;
    line-height: 1.32;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.card-artist {
    font-family: 'Inter', sans-serif;
    font-size: 0.92rem;
    font-weight: 500;
    color: #9A9A9A;
    margin: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* Featured ("now playing") card — solid tinted ground instead of a pale
   wash, so it visibly leads the stack rather than just looking faintly
   different. */
.song-card-top.featured-bg {
    background: linear-gradient(135deg, #2B2A2E 0%, #3A3340 100%);
}
.song-card-top.featured-bg .card-title {
    color: #FBF7F5;
}
.song-card-top.featured-bg .card-artist {
    color: #C8BFC9;
}

/* Player groove — same surface as the rest of the card, separated only by
   a hairline, with a small caption so it reads as a deliberate "listen"
   panel rather than a default widget dropped at the bottom. The wrapper
   is forced to respect the card's width so the native control bar (which
   has its own minimum content width) never spills past the rounded edge. */
.card-player-caption {
    font-family: 'Inter', sans-serif;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #8BAF9A;
    padding: 14px 14px 0 14px;
}
.card-player {
    box-sizing: border-box;
    width: 100%;
    margin: 6px 0 0 0;
    padding: 10px 20px 22px 20px;
    background: #FDFCFB;
    border-top: 1px solid #F3EEEA;
    border-radius: 0 0 16px 16px;
}
.card-player audio {
    display: block;
    box-sizing: border-box;
    width: 100%;
    max-width: 100%;
    min-width: 0;
    height: 42px;
}
/* disableremoteplayback + controlslist on the <audio> tag itself is what
   actually removes the cast icon and trims the overflow menu on modern
   Chrome — this pseudo-element rule is just a belt-and-suspenders fallback
   for older browser builds that still honor it. */
.card-player audio::-webkit-media-controls-overflow-button {
    display: none !important;
}

/* ── Results header ── */
.results-for {
    font-family: 'Inter', sans-serif;
    font-size: 0.98rem;
    color: #888;
    margin-bottom: 20px;
}
.results-for strong {
    color: #1C1C1E;
    font-weight: 600;
}

/* ── Next Up divider — small gold section break between the Now Playing
   card and the rest of the list. Kept simple: one long fading line on
   each side, one gold diamond flanking the italic serif label. ── */
.next-up-divider {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 10px 0 22px;
}
.next-up-divider .line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, transparent, #C9963B 50%, transparent);
}
.next-up-divider .label {
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #C9963B;
    white-space: nowrap;
    display: flex;
    align-items: center;
    gap: 10px;
}
.next-up-divider .diamond {
    font-size: 0.65rem;
    color: #C9963B;
    line-height: 1;
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
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #C9963B;
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
    width: 130px;
}
.bar-value {
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-weight: 700;
    font-size: 1.25rem;
    color: #1C1C1E;
    margin-bottom: 8px;
}
.bar-rect {
    width: 64px;
    border-radius: 10px 10px 3px 3px;
    transition: height 0.35s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
.bar-rect.personal { background: linear-gradient(180deg, #f0bdb1 0%, #E8A598 100%); }
.bar-rect.diverse  { background: linear-gradient(180deg, #d9cce7 0%, #C4B5D4 100%); }
.barchart-baseline {
    width: 100%;
    max-width: 340px;
    height: 3px;
    background: #D8D5CE;
    border-radius: 2px;
    margin: 0 auto;
}
.barchart-labels {
    display: flex;
    justify-content: center;
    gap: 56px;
    margin-top: 10px;
}
.bar-label-col {
    width: 130px;
    box-sizing: border-box;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
}
.bar-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    text-align: center;
    white-space: nowrap;
}
.bar-label.personal { color: #E8A598; }
.bar-label.diverse  { color: #C4B5D4; }
.bar-desc {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    color: #999999;
    margin-top: 2px;
    text-align: center;
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
        "Personalized ◂──────────▸  Diverse",
        min_value=1, max_value=9, value=5, step=1
    )
    content_based_weight = 1 - (diversity / 10)
    personal_pct = (10 - diversity) * 10
    diverse_pct  = diversity * 10
    max_bar_height = 110
    min_bar_height = 16
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
# Spine colors cycle through the palette so position in the list is legible
# at a glance — like a shelf of LP sleeves, not a uniform stack.
def render_card(display_index, rec, featured=False, card_key=None):
    title  = rec["name"].title()
    artist = rec["artist"].title()
    url    = rec["spotify_preview_url"]

    cycle_pos = display_index % 4

    if featured:
        label_html  = '<span class="live-dot"></span>Now playing'
        spine_class = "spine-featured"
        disc_class  = "card-disc disc-featured spin"
        top_class   = "song-card-top featured-bg"
        label_class = "card-label label-featured"
    else:
        label_html  = f"Track {display_index + 1:02d}"
        spine_class = f"spine-{cycle_pos}"
        disc_class  = f"card-disc disc-{cycle_pos}"
        top_class   = "song-card-top"
        label_class = f"card-label label-{cycle_pos}"

    # st.container(key=...) renders as a real <div data-testid="stVerticalBlock">
    # with a matching CSS class (st-key-<key>) that we can target — this keeps
    # the title HTML and the player in one DOM node. The player itself is a
    # raw <audio> tag (rather than st.audio) so we can set disableremoteplayback
    # and controlslist directly: modern Chrome's redesigned native audio
    # controls no longer respond to the old ::-webkit-media-controls-* CSS
    # hooks for the cast icon or overflow menu, but they still honor these
    # HTML attributes, which is the only thing that reliably removes them.
    with st.container(key=card_key):
        st.markdown(f"""
        <div class="{top_class} {spine_class}">
            <div class="{disc_class}"></div>
            <div class="card-body">
                <div class="{label_class}">{label_html}</div>
                <div class="card-title">{title}</div>
                <div class="card-artist">{artist}</div>
            </div>
        </div>
        <div class="card-player-caption">▸ Preview</div>
        <div class="card-player">
            <audio controls preload="none" controlslist="nodownload noplaybackrate noremoteplayback" disableremoteplayback>
                <source src="{url}" type="audio/mpeg">
            </audio>
        </div>
        """, unsafe_allow_html=True)

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
                if display_i == 1:
                    st.markdown("""
                    <div class="next-up-divider">
                        <span class="line"></span>
                        <span class="label"><span class="diamond">◆</span>Next Up<span class="diamond">◆</span></span>
                        <span class="line"></span>
                    </div>
                    """, unsafe_allow_html=True)
                render_card(display_i, rec, featured=(display_i == 0), card_key=f"content_card_{display_i}")
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
            if display_i == 1:
                st.markdown("""
                <div class="next-up-divider">
                    <span class="line"></span>
                    <span class="label"><span class="diamond">◆</span>Next Up<span class="diamond">◆</span></span>
                    <span class="line"></span>
                </div>
                """, unsafe_allow_html=True)
            render_card(display_i, rec, featured=(display_i == 0), card_key=f"hybrid_card_{display_i}")