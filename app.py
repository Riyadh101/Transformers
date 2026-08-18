"""
=============================================================================
 THE TRANSFORMER PIPELINE — An Interactive Explainer
 Unit 8, Part 5: Transformers & Large Language Models (Tuwaiq Academy)
=============================================================================

 What this app does
 ------------------
 It walks a learner through four stages of the Transformer architecture,
 using *their own typed sentence* as the running example:

   Stage 1  Input Embeddings & Positional Encoding
   Stage 2  Multi-Head Self-Attention (Encoder Block)
   Stage 3  Encoder-Decoder Cross-Attention (Decoder Block)
   Stage 4  Linear, Softmax & Output Word Prediction

 Every number on screen is really computed from the typed text — nothing is
 faked. Weight matrices are pseudo-random but seeded, so results are stable
 across reruns and reproducible for classroom demos.

 Run it with:
     streamlit run app.py

 Requirements: streamlit, numpy, pandas, plotly
=============================================================================
"""

import hashlib
import re
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# 1. PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="The Transformer Pipeline",
    page_icon="//",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Design tokens — kept in one place so the palette stays consistent everywhere,
# including inside the Plotly charts.
# -----------------------------------------------------------------------------
INK = "#05070F"        # deepest background
CYAN = "#22E0FF"       # primary neon — data, keys, "machine" side
VIOLET = "#A66BFF"     # secondary neon — queries, attention
MINT = "#5EF2C4"       # tertiary neon — values, human-readable side
AMBER = "#FFC24B"      # highlight — masks, warnings, predictions
TEXT = "#DCE6FF"       # body text
MUTED = "#8FA0C8"      # captions

# Custom Plotly colour ramps built from the same palette.
SCALE_COOL = [[0.0, "#070C22"], [0.45, "#4C2A9E"], [0.75, VIOLET], [1.0, CYAN]]
SCALE_HOT = [[0.0, "#0A0A1E"], [0.5, "#B4531E"], [1.0, AMBER]]
SCALE_MINT = [[0.0, "#04121A"], [0.5, "#1C7F80"], [1.0, MINT]]
SCALE_BITS = [[0.0, "#0B1024"], [1.0, CYAN]]


# =============================================================================
# 2. CUSTOM CSS — glassmorphism over a dark "neon tech" surface
# =============================================================================

def inject_css() -> None:
    """Load web fonts and paint the whole dashboard."""
    st.markdown(
        """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">

<style>
/* ---------- Base surface ------------------------------------------------ */
.stApp {
    background:
        radial-gradient(1100px 620px at 12% -10%, rgba(166,107,255,0.20), transparent 60%),
        radial-gradient(900px 520px at 88% 0%, rgba(34,224,255,0.16), transparent 55%),
        radial-gradient(800px 800px at 50% 120%, rgba(94,242,196,0.10), transparent 60%),
        #05070F;
    color: #DCE6FF;
    font-family: 'Inter', system-ui, sans-serif;
}
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1500px; }

h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.01em; color: #F2F6FF; }
p, li, label, span { color: #DCE6FF; }
code, kbd { font-family: 'JetBrains Mono', monospace; }

/* ---------- Masthead ---------------------------------------------------- */
.masthead {
    border: 1px solid rgba(140,180,255,0.18);
    border-radius: 22px;
    padding: 26px 30px;
    background: linear-gradient(135deg, rgba(166,107,255,0.16), rgba(34,224,255,0.08) 55%, rgba(255,255,255,0.03));
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    box-shadow: 0 24px 60px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.10);
    margin-bottom: 18px;
}
.masthead .kicker {
    font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; letter-spacing: 0.28em;
    text-transform: uppercase; color: #22E0FF; margin: 0 0 8px 0;
}
.masthead h1 { font-size: 2.35rem; margin: 0 0 6px 0; line-height: 1.1; }
.masthead .lede { color: #A8B7DC; font-size: 1.0rem; margin: 0; max-width: 78ch; }

/* ---------- Stage rail (the 01 -> 04 progress strip) -------------------- */
.rail { display: flex; gap: 10px; flex-wrap: wrap; margin: 18px 0 4px 0; }
.rail .node {
    flex: 1 1 190px; border-radius: 14px; padding: 12px 14px;
    border: 1px solid rgba(140,180,255,0.14);
    background: rgba(255,255,255,0.028);
    backdrop-filter: blur(10px);
}
.rail .node .num {
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    color: #7C8BB4; letter-spacing: 0.2em;
}
.rail .node .name { font-family: 'Space Grotesk', sans-serif; font-size: 0.94rem; color: #B9C6E8; }
.rail .node.on {
    border-color: rgba(34,224,255,0.55);
    background: linear-gradient(135deg, rgba(34,224,255,0.16), rgba(166,107,255,0.10));
    box-shadow: 0 0 26px rgba(34,224,255,0.22), inset 0 1px 0 rgba(255,255,255,0.12);
}
.rail .node.on .num { color: #22E0FF; }
.rail .node.on .name { color: #FFFFFF; }
.rail .node.done { border-color: rgba(94,242,196,0.35); }
.rail .node.done .num { color: #5EF2C4; }

/* ---------- Signature element: the live bit ribbon ---------------------- */
.ribbon {
    margin: 14px 0 22px 0; padding: 10px 14px; border-radius: 12px;
    border: 1px solid rgba(34,224,255,0.22);
    background: linear-gradient(90deg, rgba(34,224,255,0.10), rgba(5,7,15,0.6) 70%);
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; letter-spacing: 0.14em;
    color: #7FE9FF; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    text-shadow: 0 0 12px rgba(34,224,255,0.45);
}
.ribbon b { color: #5EF2C4; letter-spacing: 0.06em; }

/* ---------- Glass cards ------------------------------------------------- */
.card {
    border-radius: 18px; padding: 20px 22px; margin-bottom: 16px;
    border: 1px solid rgba(140,180,255,0.16);
    background: rgba(255,255,255,0.035);
    backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
    box-shadow: 0 16px 40px rgba(0,0,0,0.40), inset 0 1px 0 rgba(255,255,255,0.07);
}
.card h4 {
    margin: 0 0 10px 0; font-size: 0.80rem; letter-spacing: 0.20em; text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}
.card p { margin: 0 0 8px 0; font-size: 0.96rem; line-height: 1.62; color: #C6D2F0; }
.card p:last-child { margin-bottom: 0; }
.card.analogy { border-left: 3px solid #5EF2C4; }
.card.analogy h4 { color: #5EF2C4; }
.card.tech { border-left: 3px solid #22E0FF; }
.card.tech h4 { color: #22E0FF; }
.card.note { border-left: 3px solid #FFC24B; }
.card.note h4 { color: #FFC24B; }
.card.prompt { border-left: 3px solid #A66BFF; }
.card.prompt h4 { color: #A66BFF; }

/* ---------- Small metric chips ----------------------------------------- */
.chips { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 6px; }
.chip {
    border-radius: 12px; padding: 10px 14px; min-width: 120px;
    border: 1px solid rgba(140,180,255,0.16); background: rgba(255,255,255,0.03);
}
.chip .k { font-family: 'JetBrains Mono', monospace; font-size: 0.66rem; letter-spacing: 0.18em;
           text-transform: uppercase; color: #7C8BB4; }
.chip .v { font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem; color: #22E0FF; }

/* ---------- Section headings ------------------------------------------- */
.sect {
    font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; letter-spacing: 0.24em;
    text-transform: uppercase; color: #8FA0C8; margin: 26px 0 8px 0;
    border-bottom: 1px solid rgba(140,180,255,0.14); padding-bottom: 8px;
}

/* ---------- Streamlit widgets ------------------------------------------ */
section[data-testid="stSidebar"] {
    background: rgba(8,11,26,0.90);
    border-right: 1px solid rgba(140,180,255,0.14);
    backdrop-filter: blur(20px);
}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #F2F6FF; }

.stButton > button {
    width: 100%; border-radius: 12px; padding: 0.55rem 1rem;
    font-family: 'Space Grotesk', sans-serif; font-weight: 500;
    color: #EAF2FF; background: rgba(255,255,255,0.05);
    border: 1px solid rgba(140,180,255,0.28); transition: all 0.18s ease;
}
.stButton > button:hover {
    border-color: rgba(34,224,255,0.75); color: #FFFFFF;
    background: linear-gradient(135deg, rgba(34,224,255,0.20), rgba(166,107,255,0.16));
    box-shadow: 0 0 22px rgba(34,224,255,0.28);
}
.stButton > button:focus-visible { outline: 2px solid #5EF2C4; outline-offset: 2px; }
.stButton > button:disabled { opacity: 0.35; }

.stTextInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    color: #EAF2FF !important;
    border: 1px solid rgba(140,180,255,0.22) !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: rgba(34,224,255,0.7) !important; box-shadow: 0 0 16px rgba(34,224,255,0.20) !important;
}

div[data-testid="stMetricValue"] { color: #22E0FF; font-family: 'Space Grotesk', sans-serif; }

.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid rgba(140,180,255,0.14); }
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.03); border-radius: 10px 10px 0 0;
    padding: 8px 16px; font-family: 'Space Grotesk', sans-serif; color: #A8B7DC;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(34,224,255,0.18), rgba(166,107,255,0.12));
    color: #FFFFFF !important;
}

div[data-testid="stExpander"] {
    border: 1px solid rgba(140,180,255,0.16); border-radius: 14px;
    background: rgba(255,255,255,0.028); overflow: hidden;
}

.stDataFrame { border-radius: 12px; overflow: hidden; }
hr { border-color: rgba(140,180,255,0.14); }

/* Respect users who prefer less motion. */
@media (prefers-reduced-motion: reduce) {
    * { animation: none !important; transition: none !important; }
}
</style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# 3. SMALL HTML HELPERS
# =============================================================================

def card(title: str, body_html: str, kind: str = "tech") -> None:
    """Render a glass card. `kind` picks the accent: tech | analogy | note | prompt."""
    st.markdown(
        f'<div class="card {kind}"><h4>{title}</h4>{body_html}</div>',
        unsafe_allow_html=True,
    )


def section(label: str) -> None:
    """A thin lettered divider that labels the block underneath it."""
    st.markdown(f'<div class="sect">{label}</div>', unsafe_allow_html=True)


def chips(pairs) -> None:
    """A row of small key/value chips, e.g. token count or model width."""
    html = '<div class="chips">'
    for key, value in pairs:
        html += f'<div class="chip"><div class="k">{key}</div><div class="v">{value}</div></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# 4. THE MATH — tokenisation, encoding, attention
#    Every function here is pure and cached, so the app stays fast and the
#    same sentence always produces the same numbers.
# =============================================================================

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+|[^\sA-Za-z0-9']")


@st.cache_data(show_spinner=False)
def tokenize(text: str, max_tokens: int = 10) -> list:
    """Split text into word-ish tokens. Simple on purpose: the point is to make
    the token boundaries visible, not to reproduce a real BPE vocabulary."""
    tokens = TOKEN_PATTERN.findall(text.strip())
    return tokens[:max_tokens] if tokens else ["hello"]


@st.cache_data(show_spinner=False)
def token_id(token: str, vocab_size: int = 30000) -> int:
    """Deterministic pseudo token-ID, the way a real tokenizer maps text -> int."""
    digest = hashlib.md5(token.lower().encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % vocab_size


def to_bits(token: str) -> str:
    """UTF-8 bytes of a token, rendered as raw zeros and ones."""
    return "".join(f"{byte:08b}" for byte in token.encode("utf-8"))


def to_hex(token: str) -> str:
    return " ".join(f"{byte:02X}" for byte in token.encode("utf-8"))


@st.cache_data(show_spinner=False)
def embed(tokens: tuple, d_model: int) -> np.ndarray:
    """Look up a learned embedding vector per token.

    A trained model reads these from a weight table. Here each vector is drawn
    from a generator seeded by the token string itself, so the same word always
    gets the same vector — which is the property that actually matters for the
    demo (repeat words land on repeat rows)."""
    rows = []
    for token in tokens:
        seed = int(hashlib.md5(token.lower().encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        rows.append(rng.normal(0.0, 0.55, d_model))
    return np.vstack(rows)


@st.cache_data(show_spinner=False)
def positional_encoding(n_positions: int, d_model: int) -> np.ndarray:
    """Sinusoidal positional encoding from 'Attention Is All You Need'.

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    pe = np.zeros((n_positions, d_model))
    position = np.arange(n_positions)[:, None]
    even_index = np.arange(0, d_model, 2)
    angle = position / np.power(10000.0, even_index / d_model)
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle[:, : pe[:, 1::2].shape[1]])
    return pe


@st.cache_data(show_spinner=False)
def projection(d_model: int, d_head: int, seed: int, tag: str) -> np.ndarray:
    """One learned projection matrix (W_Q, W_K or W_V) for one head."""
    mix = int(hashlib.md5(f"{tag}-{seed}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(mix)
    return rng.normal(0.0, 1.0 / np.sqrt(d_model), (d_model, d_head))


def softmax(x: np.ndarray, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with an optional temperature."""
    scaled = x / max(temperature, 1e-6)
    shifted = scaled - np.max(scaled, axis=axis, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=axis, keepdims=True)


def attention(query: np.ndarray, key: np.ndarray, value: np.ndarray, mask: np.ndarray = None):
    """Scaled dot-product attention.  Attention(Q,K,V) = softmax(QKᵀ / √d) V"""
    d_head = query.shape[-1]
    scores = query @ key.T
    scaled = scores / np.sqrt(d_head)
    if mask is not None:
        scaled = scaled + mask
    weights = softmax(scaled, axis=-1)
    return scores, scaled, weights, weights @ value


def look_ahead_mask(n: int) -> np.ndarray:
    """Upper-triangular −inf mask: position t may not read positions > t."""
    mask = np.zeros((n, n))
    mask[np.triu_indices(n, k=1)] = -np.inf
    return mask


def multi_head(x: np.ndarray, d_model: int, n_heads: int, seed: int,
               context: np.ndarray = None, mask: np.ndarray = None):
    """Run every head and return the per-head weights plus the concatenated output."""
    d_head = max(d_model // n_heads, 2)
    source = x if context is None else context
    per_head_weights, per_head_out = [], []
    for head in range(n_heads):
        w_q = projection(d_model, d_head, seed, f"Q{head}")
        w_k = projection(d_model, d_head, seed, f"K{head}")
        w_v = projection(d_model, d_head, seed, f"V{head}")
        q, k, v = x @ w_q, source @ w_k, source @ w_v
        _, _, weights, out = attention(q, k, v, mask)
        per_head_weights.append(weights)
        per_head_out.append(out)
    return per_head_weights, np.hstack(per_head_out)


def layer_norm(x: np.ndarray) -> np.ndarray:
    """Add & Norm's normalisation half, per row."""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + 1e-6
    return (x - mean) / std


# =============================================================================
# 5. PLOTLY HELPERS — one dark theme, applied everywhere
# =============================================================================

def style(fig: go.Figure, height: int = 380, title: str = "") -> go.Figure:
    fig.update_layout(
        height=height,
        title=dict(text=title, font=dict(family="Space Grotesk", size=15, color=TEXT), x=0.01),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=MUTED, size=12),
        margin=dict(l=60, r=30, t=50 if title else 24, b=44),
        hoverlabel=dict(bgcolor="#0B1024", bordercolor=CYAN, font=dict(family="JetBrains Mono", color=TEXT)),
    )
    fig.update_xaxes(gridcolor="rgba(140,180,255,0.10)", zerolinecolor="rgba(140,180,255,0.18)")
    fig.update_yaxes(gridcolor="rgba(140,180,255,0.10)", zerolinecolor="rgba(140,180,255,0.18)")
    return fig


def heatmap(matrix: np.ndarray, x_labels, y_labels, title: str,
            scale=SCALE_COOL, height: int = 380, text_fmt: str = None,
            colorbar_title: str = "") -> go.Figure:
    """A labelled matrix view — the workhorse chart of this app."""
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=x_labels,
            y=y_labels,
            colorscale=scale,
            texttemplate=text_fmt,
            textfont=dict(family="JetBrains Mono", size=11, color="#E8F2FF"),
            colorbar=dict(title=dict(text=colorbar_title, font=dict(size=11)),
                          outlinewidth=0, tickfont=dict(size=10)),
            hovertemplate="row %{y}<br>col %{x}<br>value %{z:.3f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    return style(fig, height, title)


# =============================================================================
# 6. WRITTEN CONTENT — analogy, technical explanation, formula, image prompt
#    Kept as data so the same text is reusable outside the app.
# =============================================================================

SHARED_STYLE_SUFFIX = (
    "unified art direction: glassmorphism panels with frosted translucent surfaces and 1px "
    "luminous edges, floating in deep indigo-black space (#05070F); neon accent palette of "
    "electric cyan (#22E0FF), ultraviolet purple (#A66BFF), mint green (#5EF2C4) and a single "
    "amber highlight (#FFC24B); volumetric cinematic lighting, soft bloom, subtle lens haze, "
    "thin holographic grid receding into the background, shallow depth of field, 3/4 isometric "
    "camera, ultra-detailed technical illustration, editorial infographic quality, no photorealistic "
    "humans, crisp legible sans-serif labels --ar 16:9 --style raw --v 6"
)

STAGES = {
    1: {
        "name": "Embeddings & Position",
        "title": "From letters to located vectors",
        "lede": "A Transformer never sees your sentence. It sees a stack of numbers — and until we "
                "tell it otherwise, that stack has no word order at all.",
        "analogy_title": "Analogy — the library card catalogue",
        "analogy": (
            "<p>Imagine every book in a library is described by a card carrying scores for its themes: "
            "how adventurous, how romantic, how technical. Two books with similar cards sit near each "
            "other on the shelf, even if their titles share no words. That card is the <b>embedding</b>.</p>"
            "<p>But a catalogue card says nothing about where the book sits on the shelf. So the "
            "librarian stamps a shelf coordinate onto every card. That stamp is <b>positional encoding</b> — "
            "and it is why <i>“The man drove the woman to the store”</i> stops meaning the same thing as "
            "<i>“The woman drove the man to the store.”</i></p>"
        ),
        "tech": (
            "<p>Text is split into <b>tokens</b>, each mapped to an integer ID, and each ID looks up a "
            "learned vector of length <code>d_model</code>. Similar meanings end up in similar directions "
            "in that space — the classic <i>king / queen / man / woman</i> geometry from the slides.</p>"
            "<p>Because the encoder reads every token <b>in parallel</b>, it has no built-in notion of "
            "order. A sinusoidal positional signal is therefore <b>added</b> to each embedding: even "
            "dimensions get a sine, odd dimensions get a cosine, each at a different frequency. The sum "
            "of the two is what enters the first attention layer.</p>"
        ),
        "formula": r"""
\begin{aligned}
PE_{(pos,\,2i)} &= \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) \\[4pt]
PE_{(pos,\,2i+1)} &= \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) \\[6pt]
Z^{(0)} &= \text{Embedding}(x) + PE
\end{aligned}
""",
        "prompt": (
            "A cinematic 3D infographic titled INPUT EMBEDDINGS + POSITIONAL ENCODING. On the left, "
            "the glowing typed sentence \"THE CAT SAT\" fractures into three frosted-glass token capsules; "
            "beneath each capsule a cascading waterfall of cyan binary digits (0 and 1) falls into a "
            "translucent intake funnel. In the centre, each token rises as a vertical stack of glass "
            "cells filled with luminous decimal values, forming three tall vector columns lit from "
            "within by ultraviolet light. On the right, a large mint-green plus sign fuses those columns "
            "with a second set of columns generated by two elegant sine and cosine waves that ripple "
            "across a holographic grid, each wave crest tagged pos 1, pos 2, pos 3 in amber. The merged "
            "result exits as a single glowing tensor slab labelled READY FOR ATTENTION. "
            + SHARED_STYLE_SUFFIX
        ),
    },
    2: {
        "name": "Multi-Head Self-Attention",
        "title": "Every word interrogates every other word",
        "lede": "Attention answers one question, for every token at once: which parts of this input "
                "matter to me right now?",
        "analogy_title": "Analogy — the research student in the library",
        "analogy": (
            "<p>A student walks in with a question written on a slip of paper: that is the <b>Query</b>. "
            "Every shelf carries a label — Fruits, Electronics, Cleaning — and those labels are the "
            "<b>Keys</b>. The student compares the slip against every label at a glance and scores the "
            "match.</p>"
            "<p>The books actually pulled off the matching shelves are the <b>Values</b>. The student "
            "doesn't take one book and ignore the rest; they take a weighted blend — 70% from the shelf "
            "that matched best, a little from the others. Multi-head attention is simply several students "
            "sent in at once, each briefed to look for something different: one tracks grammar, another "
            "tracks who-did-what-to-whom.</p>"
        ),
        "tech": (
            "<p>Each token projects its vector three ways to produce a Query, a Key and a Value. Scoring "
            "every query against every key gives an n×n grid of raw compatibilities. Dividing by "
            "<code>√d</code> keeps those scores in a range where softmax stays sensitive rather than "
            "saturating, and softmax turns each row into weights that sum to 1.</p>"
            "<p>Those weights are then applied to the Values, so each output vector is a context-aware "
            "mixture of the whole sentence — which is how <i>“it”</i> ends up pointing at <i>“animal”</i>. "
            "Heads run in parallel, get concatenated, and pass through a linear layer; a residual "
            "connection plus layer normalisation (<b>Add &amp; Norm</b>) keeps the signal stable.</p>"
        ),
        "formula": r"""
\begin{aligned}
\text{Attention}(Q,K,V) &= \text{softmax}\!\left(\frac{QK^{T}}{\sqrt{d}}\right)V \\[6pt]
\text{head}_i &= \text{Attention}(XW_i^{Q},\, XW_i^{K},\, XW_i^{V}) \\[4pt]
\text{MultiHead}(X) &= \text{Concat}(\text{head}_1,\dots,\text{head}_h)\,W^{O}
\end{aligned}
""",
        "prompt": (
            "A cinematic 3D infographic titled MULTI-HEAD SELF-ATTENTION. Centre stage: three stacked "
            "translucent glass matrices labelled QUERY in ultraviolet, KEY in electric cyan and VALUE in "
            "mint, each a grid of softly glowing numeric cells, tilted in isometric perspective. Beams "
            "of light shoot from every Query row to every Key column, crossing to form a bright square "
            "attention grid where a few cells burn amber-hot to show high scores; the grid is annotated "
            "softmax(QKᵀ/√d). Behind the main assembly, three ghosted duplicate assemblies fade into the "
            "haze to signal parallel heads, joined by a horizontal bar labelled CONCATENATE that feeds a "
            "single glass LINEAR LAYER slab. A thin luminous arc loops around the outside of the whole "
            "block, tagged ADD &amp; NORM residual connection. "
            + SHARED_STYLE_SUFFIX
        ),
    },
    3: {
        "name": "Cross-Attention & Masking",
        "title": "The decoder writes, one token at a time",
        "lede": "The decoder is allowed to look at the entire input sentence — but only at the part of "
                "its own output it has already written.",
        "analogy_title": "Analogy — the simultaneous interpreter",
        "analogy": (
            "<p>An interpreter listens to a full sentence in English and speaks it back in French. While "
            "speaking, she constantly glances back at what the speaker said — that glance is "
            "<b>cross-attention</b>: her Queries come from the French she is producing, while the Keys "
            "and Values come from the English she heard.</p>"
            "<p>What she cannot do is quote the French words she hasn't said yet. Training a decoder "
            "without that restriction would be like an exam where the answer sheet is face-up on the "
            "desk. The <b>look-ahead mask</b> is the sheet of paper laid over the future words.</p>"
        ),
        "tech": (
            "<p>The decoder block runs two attention layers. The first is <b>masked</b> self-attention "
            "over the target sequence: before softmax, every position above the diagonal is set to −∞, "
            "so those cells come out as exactly 0 probability. That is the triangular pattern in the "
            "slides.</p>"
            "<p>The second is the <b>cross-attention</b> layer. Its Queries come from the decoder, while "
            "its Keys and Values come from the encoder's output — the sequence of context-rich "
            "embeddings the encoder produced. This is the bridge between understanding the input and "
            "generating the output.</p>"
        ),
        "formula": r"""
\begin{aligned}
M_{ij} &= \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases} \\[6pt]
\text{Masked}(Q,K,V) &= \text{softmax}\!\left(\frac{QK^{T}}{\sqrt{d}} + M\right)V \\[6pt]
\text{Cross}(Q_{dec},K_{enc},V_{enc}) &= \text{softmax}\!\left(\frac{Q_{dec}K_{enc}^{T}}{\sqrt{d}}\right)V_{enc}
\end{aligned}
""",
        "prompt": (
            "A cinematic 3D infographic titled ENCODER-DECODER CROSS-ATTENTION. Two tall frosted-glass "
            "towers face each other across dark indigo space: the left tower, rimmed in electric cyan, is "
            "labelled ENCODER and holds a horizontal band of finished context vectors at its summit; the "
            "right tower, rimmed in ultraviolet, is labelled DECODER. Two thick braided light bridges "
            "labelled K and V arc from the encoder summit into the decoder's mid-section, while a third "
            "beam labelled Q rises from inside the decoder to meet them at a glowing junction node. In "
            "the decoder's lower chamber, a square attention grid is half-covered by a translucent amber "
            "triangular shield stamped LOOK-AHEAD MASK, the blocked cells reading −inf and dimmed to "
            "near black while the lower-left cells glow with live probabilities. "
            + SHARED_STYLE_SUFFIX
        ),
    },
    4: {
        "name": "Softmax & Prediction",
        "title": "One vector becomes one word",
        "lede": "Whatever else it is doing, the model's final act is always the same: turn a vector into "
                "a probability for every word it knows, and pick one.",
        "analogy_title": "Analogy — the talent-show scoreboard",
        "analogy": (
            "<p>Every word in the vocabulary walks on stage and receives a raw score from the judges. "
            "The scores are unbounded and hard to read — one contestant might get 8.4, another −3.1. "
            "The <b>linear layer</b> is the panel of judges; those raw scores are the <b>logits</b>.</p>"
            "<p><b>Softmax</b> is the scoreboard that converts scores into percentages summing to 100%. "
            "<b>Temperature</b> is the judges' mood: cold and strict, and the favourite takes almost "
            "everything; warm and generous, and the outsiders get a real chance. Then the winner is "
            "appended to the sentence and the entire show runs again for the next word — until "
            "<code>&lt;End&gt;</code> wins.</p>"
        ),
        "tech": (
            "<p>The decoder's final hidden vector is multiplied by a linear classifier whose output width "
            "equals the vocabulary size — tens of thousands of logits, one per token. Softmax normalises "
            "them into a probability distribution.</p>"
            "<p>Greedy decoding takes the argmax; sampling draws from the distribution, which is why the "
            "same prompt can produce different text. Because the chosen token is fed back in as the next "
            "input, generation is <b>autoregressive</b>: this loop, repeated at scale over a "
            "transformer-based model trained on massive text with billions of parameters, is what an "
            "<b>LLM</b> is.</p>"
        ),
        "formula": r"""
\begin{aligned}
\text{logits} &= h_{t}W^{T} + b \in \mathbb{R}^{|V|} \\[6pt]
P(w_i \mid x) &= \frac{e^{\text{logit}_i / T}}{\sum_{j=1}^{|V|} e^{\text{logit}_j / T}} \\[6pt]
\hat{w}_t &= \arg\max_i P(w_i \mid x)
\end{aligned}
""",
        "prompt": (
            "A cinematic 3D infographic titled LINEAR + SOFTMAX OUTPUT PREDICTION. A single luminous "
            "context vector rises from below into a wide frosted-glass slab labelled LINEAR CLASSIFIER, "
            "and emerges above it as a long horizontal ribbon of thousands of tiny glass cells stretching "
            "into the distance, each cell a raw logit, tagged N CLASS = VOCAB SIZE. The ribbon passes "
            "through a curved prism labelled SOFTMAX that bends the flat spread into a glowing probability "
            "skyline of vertical bars: most bars are dim cyan stubs, one towering bar burns amber at 0.82 "
            "and is crowned by a floating word capsule reading the predicted token, with a smaller capsule "
            "beside it reading &lt;End&gt;. A thin mint feedback arrow curls from the winning word back "
            "down to the base of the image, labelled AUTOREGRESSIVE LOOP. "
            + SHARED_STYLE_SUFFIX
        ),
    },
}


# =============================================================================
# 7. LAYOUT PIECES
# =============================================================================

def masthead(step: int) -> None:
    stage = STAGES[step]
    st.markdown(
        textwrap.dedent(
            f"""
<div class="masthead">
<p class="kicker">Unit 8 · Part 5 · Stage {step:02d} of 04</p>
<h1>{stage['title']}</h1>
<p class="lede">{stage['lede']}</p>
</div>
"""
        ),
        unsafe_allow_html=True,
    )


def stage_rail(step: int) -> None:
    """The 01→04 progress strip. Order carries real meaning here: data genuinely
    flows through these stages in sequence."""
    html = '<div class="rail">'
    for index, stage in STAGES.items():
        state = "on" if index == step else ("done" if index < step else "")
        html += (
            f'<div class="node {state}"><div class="num">{index:02d}</div>'
            f'<div class="name">{stage["name"]}</div></div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def bit_ribbon(text: str, tokens: list) -> None:
    """Signature element: the live bit strip. Same sentence, seen the way the
    machine first receives it."""
    bits = "".join(to_bits(token) for token in tokens)
    preview = bits[:168] + ("…" if len(bits) > 168 else "")
    st.markdown(
        f'<div class="ribbon"><b>{text.strip()[:40]} →</b> {preview}</div>',
        unsafe_allow_html=True,
    )


def concept_block(step: int) -> None:
    """Analogy + technical explanation + formula + the matching image prompt."""
    stage = STAGES[step]
    left, right = st.columns(2, gap="medium")
    with left:
        card(stage["analogy_title"], stage["analogy"], "analogy")
    with right:
        card("Technical explanation", stage["tech"], "tech")

    section("The formula")
    st.latex(stage["formula"])

    with st.expander(f"Image generation prompt — Stage {step:02d} (DALL·E 3 / Midjourney)"):
        st.caption(
            "All four prompts share one art direction block, so the generated set reads as a single "
            "visual family. Copy the whole thing, including the trailing parameters."
        )
        st.code(stage["prompt"], language="text")


def image_placeholder(step: int) -> None:
    """Where the generated artwork drops in. Until then, an honest empty state."""
    stage = STAGES[step]
    st.markdown(
        textwrap.dedent(
            f"""
<div class="card prompt" style="text-align:center; padding:40px 22px;">
<h4>Figure slot — Stage {step:02d}</h4>
<p style="font-family:'JetBrains Mono',monospace; font-size:0.84rem; color:#8FA0C8;">
Generate the artwork with the prompt above, save it as <code>assets/stage_{step}.png</code>,
then swap this block for <code>st.image("assets/stage_{step}.png")</code>.
</p>
<p style="color:#A66BFF; font-family:'Space Grotesk',sans-serif; font-size:1.05rem; margin-top:12px;">
{stage['name']}
</p>
</div>
"""
        ),
        unsafe_allow_html=True,
    )


# =============================================================================
# 8. STAGE 1 — Text to binary to vectors
# =============================================================================

def render_stage_1(tokens, d_model, embeddings, pos_enc, combined) -> None:
    concept_block(1)
    image_placeholder(1)

    section("Interactive — watch your text become numbers")
    chips([
        ("Tokens", len(tokens)),
        ("Model width", d_model),
        ("Bits of text", sum(len(to_bits(t)) for t in tokens)),
        ("Numbers produced", len(tokens) * d_model),
    ])

    tab_bits, tab_embed, tab_pos, tab_space = st.tabs(
        ["A · Tokens & binary", "B · Embedding vectors", "C · Positional encoding", "D · Vector space"]
    )

    # --- A: raw bytes ---------------------------------------------------------
    with tab_bits:
        st.caption(
            "Before any learning happens, your text is already numbers: UTF-8 bytes. "
            "The tokenizer then groups characters and assigns each group an integer ID."
        )
        table = pd.DataFrame({
            "Position": list(range(len(tokens))),
            "Token": tokens,
            "Token ID": [token_id(t) for t in tokens],
            "Hex (UTF-8)": [to_hex(t) for t in tokens],
            "Binary": [to_bits(t)[:64] + ("…" if len(to_bits(t)) > 64 else "") for t in tokens],
        })
        st.dataframe(table, width="stretch", hide_index=True)

        # Bit matrix: each row is a token, each column one bit of its UTF-8 form.
        width = max(len(to_bits(t)) for t in tokens)
        bit_matrix = np.zeros((len(tokens), width))
        for row, token in enumerate(tokens):
            for col, bit in enumerate(to_bits(token)):
                bit_matrix[row, col] = int(bit)
        st.plotly_chart(
            heatmap(
                bit_matrix,
                [f"b{i}" for i in range(width)],
                tokens,
                "Every bit of your sentence — lit cells are 1, dark cells are 0",
                SCALE_BITS,
                height=90 + 34 * len(tokens),
                colorbar_title="bit",
            ),
            width="stretch",
        )

    # --- B: embeddings --------------------------------------------------------
    with tab_embed:
        st.caption(
            f"Each token ID indexes a learned table and comes back as a {d_model}-dimensional vector. "
            "Type the same word twice and its two rows will be identical — meaning lives in the vector, "
            "not the position."
        )
        st.plotly_chart(
            heatmap(
                embeddings,
                [f"d{i}" for i in range(d_model)],
                tokens,
                "Embedding matrix  E ∈ ℝ^(tokens × d_model)",
                SCALE_COOL,
                height=110 + 38 * len(tokens),
                text_fmt="%{z:.2f}" if d_model <= 16 and len(tokens) <= 8 else None,
                colorbar_title="value",
            ),
            width="stretch",
        )

    # --- C: positional encoding ----------------------------------------------
    with tab_pos:
        st.caption(
            "Sine on even dimensions, cosine on odd ones, each at a different frequency. "
            "Low dimensions wave slowly, high dimensions wave fast — together they give every position "
            "a unique fingerprint that the model can add straight onto the embedding."
        )
        col_a, col_b = st.columns(2, gap="medium")
        with col_a:
            st.plotly_chart(
                heatmap(pos_enc, [f"d{i}" for i in range(d_model)], [f"pos {i}" for i in range(len(tokens))],
                        "Positional encoding  PE", SCALE_MINT, height=110 + 38 * len(tokens),
                        colorbar_title="value"),
                width="stretch",
            )
        with col_b:
            st.plotly_chart(
                heatmap(combined, [f"d{i}" for i in range(d_model)], tokens,
                        "Z⁽⁰⁾ = Embedding + PE  →  into attention", SCALE_COOL,
                        height=110 + 38 * len(tokens), colorbar_title="value"),
                width="stretch",
            )

        wave = go.Figure()
        for dim, colour in zip([0, 1, 4, 5], [CYAN, VIOLET, MINT, AMBER]):
            if dim < d_model:
                fine = np.arange(0, max(len(tokens), 12), 0.05)
                curve = (np.sin(fine / (10000 ** (dim / d_model))) if dim % 2 == 0
                         else np.cos(fine / (10000 ** ((dim - 1) / d_model))))
                wave.add_trace(go.Scatter(
                    x=fine, y=curve, mode="lines", name=f"dimension {dim}",
                    line=dict(color=colour, width=2.2),
                ))
        wave.add_trace(go.Scatter(
            x=list(range(len(tokens))), y=[0] * len(tokens), mode="markers+text",
            text=tokens, textposition="top center", name="your tokens",
            marker=dict(color="#FFFFFF", size=9, line=dict(color=CYAN, width=2)),
            textfont=dict(family="JetBrains Mono", size=11, color=TEXT),
        ))
        st.plotly_chart(style(wave, 340, "The waves your token positions are sampled from"), width="stretch")

    # --- D: geometry ----------------------------------------------------------
    with tab_space:
        st.caption(
            "Your tokens live in a high-dimensional space. Projected down to three axes by SVD, "
            "you can see which of your words the model considers neighbours."
        )
        centred = combined - combined.mean(axis=0, keepdims=True)
        _, _, right_vectors = np.linalg.svd(centred, full_matrices=False)
        coords = centred @ right_vectors[:3].T
        if coords.shape[1] < 3:
            coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))

        scatter = go.Figure(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers+text", text=tokens, textposition="top center",
            marker=dict(size=9, color=np.arange(len(tokens)), colorscale=SCALE_COOL,
                        line=dict(color=CYAN, width=1), opacity=0.95),
            textfont=dict(family="JetBrains Mono", size=12, color=TEXT),
            hovertemplate="%{text}<extra></extra>",
        ))
        axis_style = dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(140,180,255,0.14)",
                          zerolinecolor="rgba(140,180,255,0.22)", color=MUTED)
        scatter.update_layout(scene=dict(xaxis=dict(title="component 1", **axis_style),
                                         yaxis=dict(title="component 2", **axis_style),
                                         zaxis=dict(title="component 3", **axis_style)))
        st.plotly_chart(style(scatter, 520, "Your sentence in 3D (SVD projection of Z⁽⁰⁾)"), width="stretch")


# =============================================================================
# 9. STAGE 2 — Self-attention inside the encoder
# =============================================================================

def render_stage_2(tokens, d_model, n_heads, seed, combined) -> None:
    concept_block(2)
    image_placeholder(2)

    section("Interactive — the Q, K, V machinery on your sentence")

    d_head = max(d_model // n_heads, 2)
    head = st.slider("Inspect head", 1, n_heads, 1,
                     help="Heads are trained to specialise. Compare them — the patterns differ.") - 1

    w_q = projection(d_model, d_head, seed, f"Q{head}")
    w_k = projection(d_model, d_head, seed, f"K{head}")
    w_v = projection(d_model, d_head, seed, f"V{head}")
    query, key, value = combined @ w_q, combined @ w_k, combined @ w_v
    scores, scaled, weights, output = attention(query, key, value)

    chips([
        ("Heads", n_heads),
        ("Head width", d_head),
        ("Scale factor", f"√{d_head} = {np.sqrt(d_head):.2f}"),
        ("Row sum after softmax", "1.00"),
    ])

    tab_qkv, tab_scores, tab_weights, tab_out = st.tabs(
        ["A · Q, K, V projections", "B · Scores & scaling", "C · Attention weights", "D · Output & Add&Norm"]
    )

    # --- A --------------------------------------------------------------------
    with tab_qkv:
        st.caption(
            "One vector per token becomes three. The Query asks, the Key advertises, the Value carries "
            "the payload that actually gets mixed."
        )
        col_q, col_k, col_v = st.columns(3, gap="small")
        head_axis = [f"h{i}" for i in range(d_head)]
        for column, matrix, name, scale in [
            (col_q, query, "Q — what am I looking for?", SCALE_COOL),
            (col_k, key, "K — what do I contain?", SCALE_MINT),
            (col_v, value, "V — what do I pass on?", SCALE_HOT),
        ]:
            with column:
                st.plotly_chart(
                    heatmap(matrix, head_axis, tokens, name, scale,
                            height=110 + 36 * len(tokens), text_fmt="%{z:.2f}"),
                    width="stretch",
                )

    # --- B --------------------------------------------------------------------
    with tab_scores:
        st.caption(
            "Q·Kᵀ compares every token with every token. Dividing by √d shrinks the spread so softmax "
            "stays responsive instead of collapsing onto a single cell."
        )
        col_raw, col_scaled = st.columns(2, gap="medium")
        with col_raw:
            st.plotly_chart(
                heatmap(scores, tokens, tokens, "Raw scores  QKᵀ", SCALE_HOT,
                        height=140 + 40 * len(tokens), text_fmt="%{z:.1f}"),
                width="stretch",
            )
        with col_scaled:
            st.plotly_chart(
                heatmap(scaled, tokens, tokens, f"Scaled  QKᵀ / √{d_head}", SCALE_COOL,
                        height=140 + 40 * len(tokens), text_fmt="%{z:.2f}"),
                width="stretch",
            )
        st.markdown(
            f'<div class="card note"><h4>Why √d matters</h4><p>Raw scores here span '
            f'<b>{scores.min():.2f} → {scores.max():.2f}</b>. After scaling they span '
            f'<b>{scaled.min():.2f} → {scaled.max():.2f}</b>. Wider spreads push softmax towards a '
            f'one-hot output, and a one-hot output has almost no gradient to learn from.</p></div>',
            unsafe_allow_html=True,
        )

    # --- C --------------------------------------------------------------------
    with tab_weights:
        st.caption(
            "Read a row: “when the model processes this token, how much of its attention goes to each "
            "other token?” Every row sums to 1."
        )
        st.plotly_chart(
            heatmap(weights, tokens, tokens, f"Attention weights — head {head + 1}", SCALE_COOL,
                    height=150 + 42 * len(tokens), text_fmt="%{z:.2f}", colorbar_title="weight"),
            width="stretch",
        )

        focus = st.selectbox("Break down one token's attention", tokens, index=min(1, len(tokens) - 1))
        row = weights[tokens.index(focus)]
        bars = go.Figure(go.Bar(
            x=tokens, y=row,
            marker=dict(color=row, colorscale=SCALE_COOL, line=dict(color=CYAN, width=1)),
            text=[f"{value:.0%}" for value in row], textposition="outside",
            textfont=dict(family="JetBrains Mono", color=TEXT),
        ))
        bars.update_yaxes(range=[0, min(1.0, row.max() * 1.35)], title="attention weight")
        st.plotly_chart(style(bars, 340, f"Where “{focus}” looks"), width="stretch")

        st.markdown(
            f'<div class="card note"><h4>Reading this head</h4><p>In head {head + 1}, '
            f'<b>“{focus}”</b> sends most of its attention to <b>“{tokens[int(np.argmax(row))]}”</b> '
            f'({row.max():.0%}). Switch heads above and the answer will usually change — that is the '
            f'whole argument for having more than one.</p></div>',
            unsafe_allow_html=True,
        )

    # --- D --------------------------------------------------------------------
    with tab_out:
        st.caption(
            "The weights are applied to V, all heads are concatenated, and the result is added back to "
            "the input before normalisation. The residual path is what lets deep stacks train at all."
        )
        all_weights, concatenated = multi_head(combined, d_model, n_heads, seed)
        # The residual branch must match the concatenated width before it can be added.
        width = concatenated.shape[1]
        residual = np.pad(combined, ((0, 0), (0, max(0, width - d_model))))[:, :width]
        normalised = layer_norm(concatenated + residual)

        col_left, col_right = st.columns(2, gap="medium")
        with col_left:
            st.plotly_chart(
                heatmap(concatenated, [f"c{i}" for i in range(concatenated.shape[1])], tokens,
                        "Concat(head₁ … head_h)", SCALE_HOT, height=110 + 38 * len(tokens)),
                width="stretch",
            )
        with col_right:
            st.plotly_chart(
                heatmap(normalised, [f"c{i}" for i in range(normalised.shape[1])], tokens,
                        "After Add & Norm — encoder output", SCALE_COOL, height=110 + 38 * len(tokens)),
                width="stretch",
            )

        st.caption("All heads side by side — same sentence, different specialisations.")
        head_columns = st.columns(min(n_heads, 4), gap="small")
        for index, head_weights in enumerate(all_weights[:4]):
            with head_columns[index % len(head_columns)]:
                st.plotly_chart(
                    heatmap(head_weights, tokens, tokens, f"head {index + 1}", SCALE_COOL,
                            height=90 + 34 * len(tokens)),
                    width="stretch",
                )


# =============================================================================
# 10. STAGE 3 — Masked self-attention and cross-attention
# =============================================================================

def render_stage_3(tokens, d_model, n_heads, seed, combined) -> None:
    concept_block(3)
    image_placeholder(3)

    section("Interactive — masking, then the bridge from encoder to decoder")

    target_text = st.text_input(
        "Target sequence the decoder is producing",
        value="I am fine",
        help="In the slides this is the translation being generated. A <start> token is prepended for you.",
    )
    target_tokens = ["<start>"] + tokenize(target_text, max_tokens=6)

    d_head = max(d_model // n_heads, 2)
    target_embeddings = embed(tuple(target_tokens), d_model) + positional_encoding(len(target_tokens), d_model)

    w_q = projection(d_model, d_head, seed, "Qdec")
    w_k = projection(d_model, d_head, seed, "Kdec")
    w_v = projection(d_model, d_head, seed, "Vdec")
    q_dec, k_dec, v_dec = target_embeddings @ w_q, target_embeddings @ w_k, target_embeddings @ w_v

    mask = look_ahead_mask(len(target_tokens))
    _, masked_scores, masked_weights, _decoder_state = attention(q_dec, k_dec, v_dec, mask)
    _, _, open_weights, _ = attention(q_dec, k_dec, v_dec)

    tab_mask, tab_cross = st.tabs(["A · Masked self-attention", "B · Encoder–decoder cross-attention"])

    with tab_mask:
        st.caption(
            "Left: scores with −∞ written into every future cell. Right: after softmax those cells are "
            "exactly 0, so position 1 sees only itself, position 2 sees two tokens, and so on."
        )
        col_a, col_b = st.columns(2, gap="medium")
        display_scores = np.where(np.isinf(masked_scores), np.nan, masked_scores)
        with col_a:
            st.plotly_chart(
                heatmap(display_scores, target_tokens, target_tokens,
                        "Scaled scores + look-ahead mask  (blank = −∞)", SCALE_HOT,
                        height=150 + 42 * len(target_tokens), text_fmt="%{z:.2f}"),
                width="stretch",
            )
        with col_b:
            st.plotly_chart(
                heatmap(masked_weights, target_tokens, target_tokens,
                        "After softmax — the causal triangle", SCALE_COOL,
                        height=150 + 42 * len(target_tokens), text_fmt="%{z:.2f}"),
                width="stretch",
            )

        show_open = st.toggle("Show what happens without the mask", value=False)
        if show_open:
            st.plotly_chart(
                heatmap(open_weights, target_tokens, target_tokens,
                        "Unmasked — the decoder can read its own future (this is cheating)", SCALE_HOT,
                        height=150 + 42 * len(target_tokens), text_fmt="%{z:.2f}"),
                width="stretch",
            )
            st.markdown(
                '<div class="card note"><h4>Why this breaks training</h4><p>With the upper triangle open, '
                'predicting the next word only requires copying it from the input. Training loss drops, '
                'and the model learns nothing that survives contact with real generation, where the '
                'future genuinely does not exist yet.</p></div>',
                unsafe_allow_html=True,
            )

    with tab_cross:
        st.caption(
            "Queries come from the decoder, Keys and Values from the encoder. Rows are the words being "
            "written; columns are the words being read."
        )
        _, encoder_output = multi_head(combined, d_model, n_heads, seed)
        padded = np.pad(encoder_output, ((0, 0), (0, max(0, d_model - encoder_output.shape[1]))))[:, :d_model]

        w_q_cross = projection(d_model, d_head, seed, "Qcross")
        w_k_cross = projection(d_model, d_head, seed, "Kcross")
        w_v_cross = projection(d_model, d_head, seed, "Vcross")
        # Q comes from the decoder side, K and V from the encoder's finished output.
        q_cross = target_embeddings @ w_q_cross
        k_cross, v_cross = padded @ w_k_cross, padded @ w_v_cross
        _, _, cross_weights, _ = attention(q_cross, k_cross, v_cross)

        st.plotly_chart(
            heatmap(cross_weights, tokens, target_tokens,
                    "Cross-attention — each output word against the whole input", SCALE_MINT,
                    height=150 + 44 * len(target_tokens), text_fmt="%{z:.2f}", colorbar_title="weight"),
            width="stretch",
        )

        alignment = pd.DataFrame({
            "Output token": target_tokens,
            "Reads most from": [tokens[int(np.argmax(row))] for row in cross_weights],
            "Weight": [f"{row.max():.0%}" for row in cross_weights],
        })
        st.dataframe(alignment, width="stretch", hide_index=True)
        st.caption(
            "With untrained weights this alignment is arbitrary. In a trained translation model this "
            "table is where you would see the source word each output word was drawn from."
        )


# =============================================================================
# 11. STAGE 4 — Logits, softmax, prediction
# =============================================================================

VOCAB_EXTRAS = [
    "the", "a", "is", "are", "was", "model", "attention", "language", "learning",
    "transformer", "word", "token", "context", "vector", "and", "to", "of", "<End>",
]


def render_stage_4(tokens, d_model, n_heads, seed, combined) -> None:
    concept_block(4)
    image_placeholder(4)

    section("Interactive — from one vector to one word")

    vocabulary = list(dict.fromkeys([t.lower() for t in tokens] + VOCAB_EXTRAS))
    _, encoder_output = multi_head(combined, d_model, n_heads, seed)
    hidden = layer_norm(encoder_output)[-1]

    # Linear classifier: hidden state · vocabulary embedding matrix -> one logit per word.
    vocab_matrix = embed(tuple(vocabulary), hidden.shape[0])
    logits = vocab_matrix @ hidden

    col_temp, col_mode = st.columns([2, 1], gap="medium")
    with col_temp:
        temperature = st.slider(
            "Temperature", 0.10, 2.00, 0.80, 0.05,
            help="Below 1 sharpens the distribution towards the favourite. Above 1 flattens it.",
        )
    with col_mode:
        top_k = st.slider("Show top-k", 5, min(20, len(vocabulary)), min(10, len(vocabulary)))

    probabilities = softmax(logits, temperature=temperature)
    order = np.argsort(probabilities)[::-1][:top_k]
    top_words = [vocabulary[i] for i in order]
    top_probs = probabilities[order]

    chips([
        ("Vocabulary", len(vocabulary)),
        ("Predicted token", top_words[0]),
        ("Confidence", f"{top_probs[0]:.1%}"),
        ("Entropy", f"{-np.sum(probabilities * np.log(probabilities + 1e-12)):.2f} nats"),
    ])

    tab_logits, tab_probs = st.tabs(["A · Raw logits", "B · Softmax distribution"])

    with tab_logits:
        st.caption(
            "The linear layer produces one unbounded score per vocabulary entry. These are not "
            "probabilities — they can be negative and they do not sum to anything in particular."
        )
        logit_order = np.argsort(logits)[::-1][:top_k]
        logit_fig = go.Figure(go.Bar(
            x=[vocabulary[i] for i in logit_order], y=logits[logit_order],
            marker=dict(color=logits[logit_order], colorscale=SCALE_HOT, line=dict(color=AMBER, width=1)),
            text=[f"{logits[i]:.2f}" for i in logit_order], textposition="outside",
            textfont=dict(family="JetBrains Mono", color=TEXT),
        ))
        logit_fig.update_yaxes(title="logit")
        st.plotly_chart(style(logit_fig, 380, "Raw logits from the linear classifier"), width="stretch")

    with tab_probs:
        st.caption("Softmax exponentiates, then normalises. Now every bar is a probability and the full "
                   "vocabulary sums to 1.")
        prob_fig = go.Figure(go.Bar(
            x=top_words, y=top_probs,
            marker=dict(color=[AMBER if i == 0 else CYAN for i in range(len(top_words))],
                        line=dict(color="rgba(255,255,255,0.25)", width=1)),
            text=[f"{p:.1%}" for p in top_probs], textposition="outside",
            textfont=dict(family="JetBrains Mono", color=TEXT),
        ))
        prob_fig.update_yaxes(range=[0, min(1.0, top_probs[0] * 1.4)], title="probability")
        st.plotly_chart(style(prob_fig, 380, f"P(next word) at temperature {temperature:.2f}"), width="stretch")

        st.dataframe(
            pd.DataFrame({
                "Rank": range(1, len(top_words) + 1),
                "Token": top_words,
                "Logit": [f"{logits[i]:.3f}" for i in order],
                "Probability": [f"{p:.2%}" for p in top_probs],
            }),
            width="stretch", hide_index=True,
        )

    st.markdown(
        f'<div class="card note"><h4>The autoregressive loop</h4>'
        f'<p>Greedy decoding would append <b>“{top_words[0]}”</b> to the sequence and run the entire '
        f'decoder again, now with one more token of context. Sampling would instead draw from this '
        f'distribution — the reason the same prompt can give you different text twice.</p>'
        f'<p>Drag temperature to 0.10 and the top token takes nearly everything: repetitive, safe output. '
        f'Push it to 2.00 and the tail lifts: more surprising, less reliable. The loop stops when '
        f'<code>&lt;End&gt;</code> wins the vote.</p></div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# 12. MAIN
# =============================================================================

def main() -> None:
    inject_css()

    if "step" not in st.session_state:
        st.session_state.step = 1

    # ---------------- Sidebar: input and controls ----------------------------
    with st.sidebar:
        st.markdown("### The Transformer Pipeline")
        st.caption("Unit 8 · Part 5 — Transformers & Large Language Models")

        st.markdown("---")
        st.markdown("#### Your sentence")
        text = st.text_area(
            "Type anything — every number in this app is computed from it",
            value="The animal didn't cross the street because it was tired",
            height=96,
        )

        st.markdown("#### Model settings")
        d_model = st.select_slider("Embedding width (d_model)", options=[8, 12, 16, 24, 32], value=16)
        n_heads = st.select_slider("Attention heads", options=[1, 2, 4, 8], value=4)
        seed = st.number_input("Weight seed", 0, 9999, 42, help="Changes the pseudo-random weight matrices.")
        max_tokens = st.slider("Max tokens shown", 3, 10, 8)

        st.markdown("---")
        st.markdown("#### Navigate")
        choice = st.radio(
            "Stage",
            list(STAGES.keys()),
            index=st.session_state.step - 1,
            format_func=lambda index: f"{index:02d} — {STAGES[index]['name']}",
            label_visibility="collapsed",
        )
        if choice != st.session_state.step:
            st.session_state.step = choice
            st.rerun()

        st.markdown("---")
        st.caption("Reference: Vaswani et al., *Attention Is All You Need* (arXiv:1706.03762)")

    tokens = tokenize(text, max_tokens=max_tokens)
    step = st.session_state.step

    # ---------------- Shared computation -------------------------------------
    embeddings = embed(tuple(tokens), d_model)
    pos_enc = positional_encoding(len(tokens), d_model)
    combined = embeddings + pos_enc

    # ---------------- Header --------------------------------------------------
    masthead(step)
    stage_rail(step)
    bit_ribbon(text, tokens)

    # ---------------- Stage body ---------------------------------------------
    if step == 1:
        render_stage_1(tokens, d_model, embeddings, pos_enc, combined)
    elif step == 2:
        render_stage_2(tokens, d_model, n_heads, seed, combined)
    elif step == 3:
        render_stage_3(tokens, d_model, n_heads, seed, combined)
    else:
        render_stage_4(tokens, d_model, n_heads, seed, combined)

    # ---------------- Footer navigation --------------------------------------
    st.markdown("---")
    previous_column, label_column, next_column = st.columns([1, 2, 1])
    with previous_column:
        if st.button("← Previous stage", disabled=step == 1, key="prev"):
            st.session_state.step = max(1, step - 1)
            st.rerun()
    with label_column:
        st.markdown(
            f"<p style='text-align:center; font-family:JetBrains Mono, monospace; "
            f"font-size:0.78rem; letter-spacing:0.2em; color:#8FA0C8; padding-top:8px;'>"
            f"STAGE {step:02d} / 04 · {STAGES[step]['name'].upper()}</p>",
            unsafe_allow_html=True,
        )
    with next_column:
        if st.button("Next stage →", disabled=step == 4, key="next"):
            st.session_state.step = min(4, step + 1)
            st.rerun()


if __name__ == "__main__":
    main()
