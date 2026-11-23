"""Streamlit Web Application: Explainable Hybrid Recommender Demo.

Provides interactive recommendation exploration, user rating histories,
transparent structured explanations, model comparisons, dietary & allergen filtering,
and pantry kitchen mode.

Run via:
    streamlit run app/app.py
    or:
    python -m src.run_app
"""

from __future__ import annotations

import html
import sys
from collections import Counter
from pathlib import Path

# Add project root and app directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

import streamlit as st

# ---------------------------------------------------------------------------
# Page Configuration (Must be very first Streamlit command)
# ---------------------------------------------------------------------------
try:
    st.set_page_config(
        page_title="Explainable Hybrid Recommender",
        page_icon="🍳",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

import pandas as pd

try:
    from app.helpers import (
        get_adaptive_alpha_explanation,
        get_cached_data,
        get_cached_models,
        get_recommendations_for_model,
        get_user_profile,
    )
except (ImportError, ModuleNotFoundError):
    from helpers import (  # type: ignore
        get_adaptive_alpha_explanation,
        get_cached_data,
        get_cached_models,
        get_recommendations_for_model,
        get_user_profile,
    )


# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------
# Warm "paper" canvas, serif display type, monospace numerals. Two signal
# colours are used consistently everywhere: teal = collaborative filtering,
# amber = content features. The accent colour follows the dataset domain.
DOMAIN_ACCENT = {
    "foodcom": ("#C2410C", "#FFF1E8"),   # terracotta
    "movielens": ("#4F46E5", "#EEF0FF"),  # indigo
}

BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,650&family=Manrope:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

:root {
  --bg: #F6F4EF;
  --surface: #FFFFFF;
  --ink: #16181D;
  --ink-2: #3A3E46;
  --muted: #6B6F76;
  --line: #E6E2D9;
  --line-2: #EFECE5;
  --cf: #0E7490;
  --cf-soft: #E3F3F6;
  --ct: #B45309;
  --ct-soft: #FBF0E2;
  --star: #E0A100;
  --good: #15803D;
  --radius: 14px;
  --shadow: 0 1px 2px rgba(22,24,29,.04), 0 6px 20px -8px rgba(22,24,29,.10);
}

html, body, .stApp, [data-testid="stSidebar"], button, input, textarea, select {
  font-family: 'Manrope', system-ui, sans-serif;
}
.stApp { background: var(--bg); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppDeployButton"], [data-testid="stDecoration"] { display: none; }
.block-container, [data-testid="stMainBlockContainer"] {
  padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1280px;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
  background: var(--surface);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 1.2rem; }
.brand { display: flex; gap: 12px; align-items: center; margin: 0 0 18px; }
.brand-mark {
  width: 40px; height: 40px; border-radius: 11px; display: grid; place-items: center;
  background: var(--ink); color: #fff; font-size: 20px; flex: none;
}
.brand-name { font-family: 'Fraunces', serif; font-weight: 650; font-size: 18px; line-height: 1.1; color: var(--ink); }
.brand-sub { font-size: 12px; color: var(--muted); margin-top: 2px; }
.side-label {
  font-size: 11px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
  color: var(--muted); margin: 18px 0 6px;
}
.side-uid {
  display: flex; justify-content: space-between; align-items: center;
  background: var(--bg); border: 1px solid var(--line); border-radius: 10px;
  padding: 8px 12px; font-size: 13px; color: var(--ink-2); margin-top: 4px;
}
.side-uid b { font-family: 'JetBrains Mono', monospace; color: var(--ink); }

/* ---------- Hero ---------- */
.hero {
  position: relative; overflow: hidden;
  background: var(--surface); border: 1px solid var(--line); border-radius: 20px;
  padding: 30px 34px 26px; box-shadow: var(--shadow); margin-bottom: 18px;
}
.hero::after {
  content: ""; position: absolute; right: -80px; top: -80px; width: 280px; height: 280px;
  border-radius: 50%; background: radial-gradient(circle, var(--accent-soft) 0%, transparent 70%);
}
.eyebrow {
  font-size: 11.5px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase;
  color: var(--accent); display: flex; align-items: center; gap: 8px;
}
.eyebrow::before { content: ""; width: 18px; height: 2px; background: var(--accent); border-radius: 2px; }
.hero h1 {
  font-family: 'Fraunces', serif !important; font-weight: 650 !important;
  font-size: clamp(28px, 3.4vw, 42px) !important; line-height: 1.08 !important;
  letter-spacing: -.02em; color: var(--ink) !important; margin: 10px 0 8px !important; padding: 0 !important;
  max-width: 820px; position: relative; z-index: 1;
}
.hero h1 em { font-style: italic; color: var(--accent); }
.hero p.lede { color: var(--muted); font-size: 15.5px; max-width: 700px; margin: 0 0 16px; position: relative; z-index: 1; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; position: relative; z-index: 1; }
.chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 11px; border-radius: 999px; font-size: 12.5px; font-weight: 600;
  background: var(--bg); color: var(--ink-2); border: 1px solid var(--line);
}
.chip.accent { background: var(--accent-soft); color: var(--accent); border-color: transparent; }
.chip.filter { background: #F1F5F0; color: var(--good); border-color: transparent; }

/* ---------- Stat cards ---------- */
.stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 8px; }
@media (max-width: 900px) { .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.stat {
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 16px 18px; box-shadow: var(--shadow);
}
.stat-k { font-size: 12px; font-weight: 600; color: var(--muted); }
.stat-v {
  font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 26px;
  color: var(--ink); letter-spacing: -.02em; margin-top: 4px;
}
.stat-v small { font-size: 13px; color: var(--muted); font-weight: 500; }
.stat-hint { font-size: 12px; color: var(--muted); margin-top: 2px; }

/* ---------- Blend (alpha) bar ---------- */
.blend { margin-top: 10px; }
.blend-bar { display: flex; height: 10px; border-radius: 999px; overflow: hidden; background: var(--line-2); }
.blend-cf { background: var(--cf); }
.blend-ct { background: var(--ct); }
.blend-legend { display: flex; justify-content: space-between; font-size: 12px; margin-top: 6px; font-weight: 600; }
.blend-legend .cf { color: var(--cf); }
.blend-legend .ct { color: var(--ct); }

/* ---------- Tabs ---------- */
.stTabs [role="tablist"], .stTabs [data-baseweb="tab-list"] {
  display: inline-flex; width: auto; gap: 4px; background: var(--surface); border: 1px solid var(--line);
  border-radius: 12px; padding: 5px; box-shadow: var(--shadow); margin: 14px 0 18px;
}
.stTabs [role="tab"] {
  height: 40px; padding: 0 18px; border-radius: 9px; font-weight: 600; color: var(--muted);
  display: flex; align-items: center; border: none !important; cursor: pointer;
  transition: background .15s ease, color .15s ease;
}
.stTabs [role="tab"]:hover { background: var(--bg); color: var(--ink); }
.stTabs [role="tab"] p { font-size: 14px; margin: 0; }
.stTabs [role="tab"][aria-selected="true"] { background: var(--ink); color: #fff !important; }
.stTabs [role="tab"][aria-selected="true"] p { color: #fff !important; }
.stTabs [role="tab"]::after, .stTabs [role="tab"]::before { display: none !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"],
.stTabs .react-aria-SelectionIndicator { display: none !important; }

/* ---------- Section headings ---------- */
.sec { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin: 4px 0 14px; flex-wrap: wrap; }
.sec h3 {
  font-family: 'Fraunces', serif !important; font-weight: 650 !important; font-size: 24px !important;
  color: var(--ink) !important; margin: 0 !important; padding: 0 !important; letter-spacing: -.01em;
}
.sec span { font-size: 13px; color: var(--muted); }

/* ---------- Recommendation cards ---------- */
[class*="st-key-rec_"] {
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 18px 20px 6px; box-shadow: var(--shadow); gap: 8px;
  transition: border-color .15s ease, transform .15s ease;
}
[class*="st-key-rec_"]:hover { border-color: #D3CDC0; transform: translateY(-1px); }
.rc-top { display: flex; justify-content: space-between; align-items: center; gap: 10px; }
.rc-rank {
  font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 13px;
  color: var(--accent); background: var(--accent-soft); border-radius: 8px; padding: 3px 9px;
}
.rc-score { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--muted); }
.rc-title {
  font-family: 'Fraunces', serif; font-weight: 650; font-size: 19px; line-height: 1.25;
  color: var(--ink); margin: 10px 0 6px; min-height: 48px;
}
.rc-rating { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--muted); }
.rc-rating b { font-family: 'JetBrains Mono', monospace; color: var(--ink); font-size: 14px; }
.stars { position: relative; display: inline-block; color: #E4DFD3; font-size: 15px; letter-spacing: 1px; line-height: 1; }
.stars-fill { position: absolute; left: 0; top: 0; overflow: hidden; white-space: nowrap; color: var(--star); width: var(--pct); }
.tags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
.tag {
  font-size: 12px; font-weight: 600; color: var(--ink-2); background: var(--bg);
  border: 1px solid var(--line-2); border-radius: 7px; padding: 3px 8px;
}
.match { margin-top: 12px; padding: 10px 12px; border-radius: 10px; background: #F1F5F0; }
.match-top { display: flex; justify-content: space-between; font-size: 12.5px; font-weight: 700; color: var(--good); }
.match-bar { height: 6px; border-radius: 999px; background: #DCE7DA; margin: 6px 0; overflow: hidden; }
.match-bar > div { height: 100%; background: var(--good); border-radius: 999px; }
.match-miss { font-size: 12px; color: var(--ink-2); }

/* Expanders inside cards */
[class*="st-key-rec_"] [data-testid="stExpander"] details {
  border: none; border-top: 1px dashed var(--line); border-radius: 0; background: transparent;
}
[class*="st-key-rec_"] [data-testid="stExpander"] summary { padding-left: 0; padding-right: 0; }
[class*="st-key-rec_"] [data-testid="stExpander"] summary p { font-size: 13px; font-weight: 700; color: var(--ink-2); }
[data-testid="stExpander"] details { border-color: var(--line); border-radius: 12px; background: var(--surface); }

.why-sum { font-size: 13.5px; color: var(--ink-2); line-height: 1.5; margin-bottom: 10px; }
.why { display: flex; gap: 10px; padding: 8px 0; border-top: 1px solid var(--line-2); }
.why:first-of-type { border-top: none; }
.why-tag {
  flex: none; height: fit-content; font-size: 10.5px; font-weight: 700; letter-spacing: .06em;
  text-transform: uppercase; padding: 3px 7px; border-radius: 6px; white-space: nowrap;
}
.why-tag.cf { color: var(--cf); background: var(--cf-soft); }
.why-tag.ct { color: var(--ct); background: var(--ct-soft); }
.why-tag.gen { color: var(--muted); background: var(--bg); }
.why-text { font-size: 13px; color: var(--ink-2); line-height: 1.45; }
.why-w { height: 3px; border-radius: 3px; background: var(--line-2); margin-top: 6px; width: 120px; overflow: hidden; }
.why-w > div { height: 100%; background: var(--ink-2); }

/* ---------- Compare ---------- */
.cmp {
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 16px; box-shadow: var(--shadow); height: 100%;
}
.cmp.active { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft), var(--shadow); }
.cmp-h { font-family: 'Fraunces', serif; font-weight: 650; font-size: 16px; color: var(--ink); }
.cmp-d { font-size: 12px; color: var(--muted); margin: 2px 0 12px; min-height: 32px; }
.cmp-i { display: flex; gap: 10px; padding: 9px 0; border-top: 1px solid var(--line-2); }
.cmp-n { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--muted); padding-top: 1px; }
.cmp-t { font-size: 13px; font-weight: 600; color: var(--ink); line-height: 1.35; }
.cmp-s { font-family: 'JetBrains Mono', monospace; font-size: 11.5px; color: var(--muted); margin-top: 2px; }
.dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: var(--accent); margin-left: 5px; vertical-align: middle; }
.legend { font-size: 12.5px; color: var(--muted); margin: -4px 0 14px; }

/* ---------- Method ---------- */
.steps { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 12px; }
@media (max-width: 900px) { .steps { grid-template-columns: 1fr; } }
.step { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 18px; box-shadow: var(--shadow); }
.step-n { font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 600; }
.step h4 { font-family: 'Fraunces', serif !important; font-size: 18px !important; margin: 6px 0 6px !important; padding: 0 !important; color: var(--ink) !important; }
.step p { font-size: 13.5px; color: var(--ink-2); line-height: 1.5; margin: 0; }

/* ---------- Panels & misc ---------- */
.panel { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 18px 20px; box-shadow: var(--shadow); }
.panel-h { font-size: 12px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }
.panel p { font-size: 14px; color: var(--ink-2); line-height: 1.55; margin: 0; }
.dist { display: grid; grid-template-columns: 26px 1fr 40px; align-items: center; gap: 8px; font-size: 12.5px; margin: 5px 0; }
.dist-bar { height: 8px; border-radius: 999px; background: var(--line-2); overflow: hidden; }
.dist-bar > div { height: 100%; background: var(--star); border-radius: 999px; }
.dist .n { font-family: 'JetBrains Mono', monospace; color: var(--muted); text-align: right; }
.empty {
  text-align: center; padding: 48px 20px; background: var(--surface); border: 1px dashed var(--line);
  border-radius: var(--radius); color: var(--muted);
}
.empty b { display: block; font-family: 'Fraunces', serif; font-size: 20px; color: var(--ink); margin-bottom: 4px; }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Small HTML helpers (strings are joined without indentation so Markdown
# never mistakes them for code blocks)
# ---------------------------------------------------------------------------
def esc(value) -> str:
    return html.escape(str(value))


def stars_html(rating: float) -> str:
    pct = max(0.0, min(1.0, float(rating) / 5.0)) * 100
    return (
        f'<span class="stars">★★★★★'
        f'<span class="stars-fill" style="--pct:{pct:.0f}%">★★★★★</span></span>'
    )


def blend_html(alpha: float) -> str:
    cf = int(round(alpha * 100))
    return (
        '<div class="blend">'
        f'<div class="blend-bar"><div class="blend-cf" style="width:{cf}%"></div>'
        f'<div class="blend-ct" style="width:{100 - cf}%"></div></div>'
        f'<div class="blend-legend"><span class="cf">Collaborative {cf}%</span>'
        f'<span class="ct">Content {100 - cf}%</span></div></div>'
    )


def clean_number(value):
    try:
        if value is None or value == "N/A" or pd.isna(value):
            return None
        return int(float(value))
    except (ValueError, TypeError):
        return None


REASON_LABELS = {
    "cf_user": ("Similar users", "cf"),
    "cf_item": ("You liked", "cf"),
    "cf_secondary": ("Community", "cf"),
    "content": ("Content match", "ct"),
    "content_secondary": ("Content signal", "ct"),
    "generic": ("Overview", "gen"),
    "fallback": ("Baseline", "gen"),
}


def section(title: str, note: str = "") -> None:
    st.markdown(
        f'<div class="sec"><h3>{esc(title)}</h3><span>{esc(note)}</span></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar: Configuration & Controls
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    '<div class="brand"><div class="brand-mark">✦</div><div>'
    '<div class="brand-name">Hybrid Recommender</div>'
    '<div class="brand-sub">Explainable · Adaptive · Transparent</div></div></div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div class="side-label">Domain</div>', unsafe_allow_html=True)
dataset_choice = st.sidebar.selectbox(
    "Select Dataset Domain:",
    options=["Food.com Recipes (Primary)", "MovieLens 100K (Benchmark)"],
    index=0,
    key="sb_dataset_domain",
    label_visibility="collapsed",
)
dataset_key = "foodcom" if "Food.com" in dataset_choice else "movielens"
accent, accent_soft = DOMAIN_ACCENT[dataset_key]
st.markdown(
    f"<style>:root{{--accent:{accent};--accent-soft:{accent_soft};}}</style>",
    unsafe_allow_html=True,
)

with st.spinner("Loading models and catalog data from pipeline cache..."):
    data = get_cached_data(dataset_key)
    models = get_cached_models(dataset_key)

cfg = data["cfg"]
items_df = data["items_df"]
train_val_df = data["train_val_df"]
name_map = data["name_map"]
user_counts = data["user_counts"]
archetypes = data["archetypes"]
all_users = data["all_users"]

# User Selector
st.sidebar.markdown('<div class="side-label">User</div>', unsafe_allow_html=True)
archetype_options = ["Custom User ID..."] + list(archetypes.keys())
selected_archetype = st.sidebar.selectbox(
    "Preset User Archetypes:",
    options=archetype_options,
    index=1,  # Default to low-activity user
    help="Archetypes chosen deterministically based on empirical activity quantiles.",
    key=f"sb_archetype_{dataset_key}",
    label_visibility="collapsed",
)

if selected_archetype == "Custom User ID...":
    default_uid = all_users[0] if all_users else 1
    user_id = int(st.sidebar.number_input(
        "Enter User ID:",
        min_value=1,
        max_value=max(all_users) if all_users else 1000000,
        value=default_uid,
        step=1,
        key=f"num_custom_user_{dataset_key}",
    ))
else:
    user_id = archetypes[selected_archetype]
    st.sidebar.markdown(
        f'<div class="side-uid">Selected user <b>#{user_id}</b></div>',
        unsafe_allow_html=True,
    )

# Model Selector
st.sidebar.markdown('<div class="side-label">Model</div>', unsafe_allow_html=True)
model_options = {
    "Adaptive Hybrid (Recommended)": "hybrid_adaptive",
    "Fixed Hybrid (50/50 Blend)": "hybrid_fixed",
    "Collaborative Filtering Only": "hybrid_cf_only",
    "Content-Based Only": "hybrid_content_only",
    "Popularity Baseline": "popularity",
}
selected_model_name = st.sidebar.selectbox(
    "Active Model:",
    options=list(model_options.keys()),
    index=0,
    key="sb_active_model",
    label_visibility="collapsed",
)
active_model_key = model_options[selected_model_name]
active_model = models[active_model_key]

top_n = st.sidebar.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10,
    step=1,
    key="sld_top_n",
)

# Recipe Options (Food.com only)
filters_dict = {}
kitchen_input = ""

if dataset_key == "foodcom":
    st.sidebar.markdown('<div class="side-label">Recipe filters</div>', unsafe_allow_html=True)
    with st.sidebar.expander("🥗 Diet, time & calories", expanded=False):
        diet_choice = st.selectbox(
            "Dietary Restriction:",
            options=["None", "Vegetarian", "Vegan", "Gluten-Free", "Nut-Free", "Dairy-Free"],
            index=0,
            key="sb_recipe_diet",
        )
        if diet_choice != "None":
            filters_dict["diet"] = diet_choice.lower().replace("-", "_")

        limit_time = st.toggle("Limit cooking time", value=False, key="cb_limit_time")
        max_time_val = st.slider(
            "Max Time (mins):",
            min_value=15, max_value=240, value=60, step=15,
            disabled=not limit_time, key="sld_max_time",
        )
        if limit_time:
            filters_dict["max_minutes"] = max_time_val

        limit_cal = st.toggle("Limit calories", value=False, key="cb_limit_cal")
        max_cal_val = st.slider(
            "Max Calories (kcal):",
            min_value=100, max_value=1500, value=600, step=50,
            disabled=not limit_cal, key="sld_max_cal",
        )
        if limit_cal:
            filters_dict["max_calories"] = max_cal_val

        disliked_raw = st.text_input(
            "Disliked Ingredients (comma-separated):",
            placeholder="e.g. mushrooms, cilantro",
            key="ti_disliked_ingredients",
        )
        if disliked_raw.strip():
            disliked_list = [x.strip() for x in disliked_raw.split(",") if x.strip()]
            if disliked_list:
                filters_dict["exclude_ingredients"] = disliked_list

    with st.sidebar.expander("🍳 Kitchen mode", expanded=False):
        kitchen_input = st.text_area(
            "Ingredients I have on hand:",
            placeholder="e.g. chicken breast, garlic, olive oil, onion, rice",
            help="Matches available ingredients against recipe lists, calculating ingredient coverage % and surfacing missing items.",
            key="ta_kitchen_ingredients",
        )

filters_to_pass = filters_dict if filters_dict else None
kitchen_to_pass = kitchen_input if kitchen_input.strip() else None

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
noun = "recipes" if dataset_key == "foodcom" else "films"
domain_label = "Food.com Recipes" if dataset_key == "foodcom" else "MovieLens 100K"
short_model = selected_model_name.split(" (")[0]

chips = [
    f'<span class="chip accent">{"🍲" if dataset_key == "foodcom" else "🎬"} {esc(domain_label)}</span>',
    f'<span class="chip">◆ {esc(short_model)}</span>',
    f'<span class="chip">User #{user_id}</span>',
    f'<span class="chip">Top {top_n}</span>',
]
if "diet" in filters_dict:
    chips.append(f'<span class="chip filter">✓ {esc(diet_choice)}</span>')
if "max_minutes" in filters_dict:
    chips.append(f'<span class="chip filter">≤ {filters_dict["max_minutes"]} min</span>')
if "max_calories" in filters_dict:
    chips.append(f'<span class="chip filter">≤ {filters_dict["max_calories"]} kcal</span>')
if "exclude_ingredients" in filters_dict:
    chips.append(f'<span class="chip filter">No {esc(", ".join(filters_dict["exclude_ingredients"]))}</span>')
if kitchen_to_pass:
    chips.append('<span class="chip filter">🍳 Kitchen mode</span>')

st.markdown(
    '<div class="hero">'
    f'<div class="eyebrow">Explainable Hybrid Recommender</div>'
    f'<h1>{noun.capitalize()} picked for user {user_id}, <em>and the reasons why.</em></h1>'
    '<p class="lede">Collaborative filtering and content similarity, blended per user by how much '
    'history they have. Every recommendation shows the evidence behind it.</p>'
    f'<div class="chips">{"".join(chips)}</div>'
    '</div>',
    unsafe_allow_html=True,
)

# Stat row
alpha_val, _alpha_text = get_adaptive_alpha_explanation(user_id, user_counts, cfg.k0)
n_u = int(user_counts.get(user_id, 0))
n_users = int(train_val_df["user_id"].nunique())
n_items = int(len(items_df))
n_ratings = int(len(train_val_df))
density = n_ratings / max(1, n_users * n_items) * 100

st.markdown(
    '<div class="stats">'
    f'<div class="stat"><div class="stat-k">Users</div><div class="stat-v">{n_users:,}</div>'
    f'<div class="stat-hint">in training history</div></div>'
    f'<div class="stat"><div class="stat-k">{noun.capitalize()}</div><div class="stat-v">{n_items:,}</div>'
    f'<div class="stat-hint">in the catalogue</div></div>'
    f'<div class="stat"><div class="stat-k">Ratings</div><div class="stat-v">{n_ratings:,}</div>'
    f'<div class="stat-hint">{density:.2f}% matrix density</div></div>'
    f'<div class="stat"><div class="stat-k">Blend for user #{user_id}</div>'
    f'<div class="stat-v">α {alpha_val:.2f} <small>· {n_u} ratings</small></div>{blend_html(alpha_val)}</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_recs, tab_profile, tab_compare, tab_method = st.tabs(
    ["Recommendations", "User profile", "Compare models", "How it works"]
)

# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------
with tab_recs:
    recs, warn = get_recommendations_for_model(
        model=active_model,
        dataset_name=dataset_key,
        user_id=user_id,
        n=top_n,
        items_df=items_df,
        name_map=name_map,
        filters=filters_to_pass,
        kitchen_ingredients=kitchen_to_pass,
    )

    section(f"Top {top_n} for user #{user_id}", f"Ranked by {short_model}")

    if warn:
        st.warning(warn)

    if not recs:
        st.markdown(
            '<div class="empty"><b>Nothing matches these settings</b>'
            'Try relaxing a filter or adding more pantry ingredients.</div>',
            unsafe_allow_html=True,
        )
    else:
        grid = st.columns(2, gap="medium")
        for idx, item in enumerate(recs, 1):
            with grid[(idx - 1) % 2]:
                with st.container(key=f"rec_{idx}"):
                    pred = float(item["predicted_rating"])

                    tags = []
                    if "category" in item and pd.notna(item["category"]):
                        tags.append(f'<span class="tag">{esc(item["category"])}</span>')
                    minutes = clean_number(item.get("total_minutes"))
                    if minutes is not None:
                        tags.append(f'<span class="tag">⏱ {minutes} min</span>')
                    kcal = clean_number(item.get("calories"))
                    if kcal is not None:
                        tags.append(f'<span class="tag">🔥 {kcal} kcal</span>')
                    if "genres" in item and pd.notna(item["genres"]):
                        for g in str(item["genres"]).replace("|", " ").split()[:4]:
                            tags.append(f'<span class="tag">{esc(g)}</span>')

                    score_txt = ""
                    if "rank_score" in item:
                        try:
                            score_txt = f"score {float(item['rank_score']):.3f}"
                        except (TypeError, ValueError):
                            score_txt = ""

                    match_html = ""
                    if "kitchen_overlap" in item:
                        pct = int(round(item["kitchen_overlap"] * 100))
                        missing = item.get("missing_ingredients") or []
                        miss_txt = (
                            "Missing: " + esc(", ".join(missing[:6])) + ("…" if len(missing) > 6 else "")
                            if missing else "You have everything you need."
                        )
                        match_html = (
                            f'<div class="match"><div class="match-top"><span>Pantry match</span>'
                            f'<span>{pct}%</span></div><div class="match-bar"><div style="width:{pct}%">'
                            f'</div></div><div class="match-miss">{miss_txt}</div></div>'
                        )

                    st.markdown(
                        f'<div class="rc-top"><span class="rc-rank">#{idx:02d}</span>'
                        f'<span class="rc-score">{score_txt}</span></div>'
                        f'<div class="rc-title">{esc(item["name"])}</div>'
                        f'<div class="rc-rating">{stars_html(pred)}<b>{pred:.2f}</b> predicted</div>'
                        + (f'<div class="tags">{"".join(tags)}</div>' if tags else "")
                        + match_html,
                        unsafe_allow_html=True,
                    )

                    with st.expander("Why this recommendation?", expanded=False):
                        struct = item.get("structured_explanation", {}) or {}
                        parts = []
                        summary_text = struct.get("summary", "")
                        if summary_text:
                            parts.append(f'<div class="why-sum">{esc(summary_text)}</div>')
                        for r in struct.get("reasons", []) or []:
                            r_text = r.get("text", "")
                            if not r_text:
                                continue
                            label, cls = REASON_LABELS.get(r.get("type", ""), ("Signal", "gen"))
                            try:
                                w = max(0.0, min(1.0, float(r.get("weight", 0))))
                            except (TypeError, ValueError):
                                w = 0.0
                            parts.append(
                                f'<div class="why"><span class="why-tag {cls}">{esc(label)}</span>'
                                f'<div><div class="why-text">{esc(r_text)}</div>'
                                f'<div class="why-w"><div style="width:{w * 100:.0f}%"></div></div></div></div>'
                            )
                        if not parts:
                            parts.append('<div class="why-sum">No detailed explanation is available for this item.</div>')
                        st.markdown("".join(parts), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# User profile
# ---------------------------------------------------------------------------
with tab_profile:
    profile_df = get_user_profile(user_id, train_val_df, items_df, name_map)
    section(f"User #{user_id}", f"{len(profile_df)} rated {noun}")

    if len(profile_df) == 0:
        st.markdown(
            f'<div class="empty"><b>No history for user #{user_id}</b>'
            'This user is unseen, so recommendations rely entirely on content features and popularity.</div>',
            unsafe_allow_html=True,
        )
    else:
        left, right = st.columns([1.15, 1], gap="medium")
        with left:
            pct_cf = int(round(alpha_val * 100))
            st.markdown(
                '<div class="panel"><div class="panel-h">How this user is scored</div>'
                f'<p>With <b>{n_u} ratings</b> and k₀ = {cfg.k0:.0f}, the adaptive weight is '
                f'<b>α = {n_u} / ({n_u} + {cfg.k0:.0f}) = {alpha_val:.2f}</b>. '
                f'So <b style="color:var(--cf)">{pct_cf}%</b> of the score comes from people with similar taste '
                f'and <b style="color:var(--ct)">{100 - pct_cf}%</b> from the item\'s own content.</p>'
                f'{blend_html(alpha_val)}</div>',
                unsafe_allow_html=True,
            )
        with right:
            counts = profile_df["Rating"].round().astype(int).value_counts()
            total = max(1, len(profile_df))
            rows = "".join(
                f'<div class="dist"><span>{s}★</span><div class="dist-bar"><div style="width:{counts.get(s, 0) / total * 100:.0f}%">'
                f'</div></div><span class="n">{counts.get(s, 0)}</span></div>'
                for s in (5, 4, 3, 2, 1)
            )
            st.markdown(
                f'<div class="panel"><div class="panel-h">Rating distribution · mean '
                f'{profile_df["Rating"].mean():.2f}</div>{rows}</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
        st.dataframe(
            profile_df,
            width="stretch",
            height=380,
            hide_index=True,
            column_config={
                "item_id": st.column_config.NumberColumn("ID", format="%d", width="small"),
                "Title / Name": st.column_config.TextColumn("Title", width="large"),
                "Rating": st.column_config.ProgressColumn(
                    "Rating", min_value=0, max_value=5, format="%d ★", width="medium"
                ),
                "Date": st.column_config.TextColumn("Rated on", width="small"),
            },
        )

# ---------------------------------------------------------------------------
# Compare models
# ---------------------------------------------------------------------------
with tab_compare:
    section("Same user, five models", "Top 5 from each, using the current filters")
    st.markdown(
        f'<div class="legend"><span class="dot"></span>&nbsp; picked by 3 or more models &nbsp;·&nbsp; '
        f'outlined column = active model</div>',
        unsafe_allow_html=True,
    )

    compare_keys = [
        ("Adaptive Hybrid", "hybrid_adaptive", "Blend weighted by user history (α)"),
        ("Fixed Hybrid", "hybrid_fixed", "Even 50/50 blend for everyone"),
        ("CF Only", "hybrid_cf_only", "Driven by similar users' ratings"),
        ("Content Only", "hybrid_content_only", "Driven by item text and features"),
        ("Popularity", "popularity", "What most people rate highly"),
    ]

    lists = {}
    for _label, key, _desc in compare_keys:
        mod = models.get(key)
        if mod is None:
            lists[key] = None
            continue
        c_recs, _ = get_recommendations_for_model(
            model=mod,
            dataset_name=dataset_key,
            user_id=user_id,
            n=5,
            items_df=items_df,
            name_map=name_map,
            filters=filters_to_pass,
            kitchen_ingredients=kitchen_to_pass,
        )
        lists[key] = c_recs

    freq = Counter(r["item_id"] for v in lists.values() if v for r in v)

    cols = st.columns(len(compare_keys), gap="small")
    for col, (c_label, key, desc) in zip(cols, compare_keys):
        c_recs = lists[key]
        if c_recs is None:
            body = '<div class="cmp-d">Model unavailable</div>'
        elif not c_recs:
            body = '<div class="cmp-d">No recommendations</div>'
        else:
            body = "".join(
                f'<div class="cmp-i"><span class="cmp-n">{i}</span><div>'
                f'<div class="cmp-t">{esc(r["name"])}{"<span class=dot></span>" if freq[r["item_id"]] >= 3 else ""}</div>'
                f'<div class="cmp-s">★ {float(r["predicted_rating"]):.2f}</div></div></div>'
                for i, r in enumerate(c_recs, 1)
            )
        active_cls = " active" if key == active_model_key else ""
        col.markdown(
            f'<div class="cmp{active_cls}"><div class="cmp-h">{esc(c_label)}</div>'
            f'<div class="cmp-d">{esc(desc)}</div>{body}</div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# How it works
# ---------------------------------------------------------------------------
with tab_method:
    section("How it works", "Three signals, one transparent score")
    content_src = (
        "recipe names, ingredients and preparation steps"
        if dataset_key == "foodcom" else "genre tokens"
    )
    st.markdown(
        '<div class="steps">'
        '<div class="step"><div class="step-n" style="color:var(--cf)">01 · COLLABORATIVE</div>'
        '<h4>People like you</h4><p>Item-based collaborative filtering with adjusted cosine similarity '
        'over mean-centred rating vectors. It finds items that are rated like the ones you loved.</p></div>'
        '<div class="step"><div class="step-n" style="color:var(--ct)">02 · CONTENT</div>'
        f'<h4>Things like this</h4><p>TF-IDF vectors over {content_src}. These work even for new users '
        'and items nobody has rated yet.</p></div>'
        '<div class="step"><div class="step-n" style="color:var(--accent)">03 · ADAPTIVE BLEND</div>'
        f'<h4>Weighted by evidence</h4><p>Each user gets their own mix: α = n / (n + k₀), with '
        f'k₀ = {cfg.k0:.0f} tuned on validation data. More history means more weight on collaborative filtering.</p></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    st.latex(r"\hat{s}(u,i) = \alpha_u \cdot s_{CF}(u,i) + (1-\alpha_u)\cdot s_{CB}(u,i), \qquad \alpha_u = \frac{n_u}{n_u + k_0}")
    if data["filter_info"]:
        fi = data["filter_info"].get("counts", {}).get("after_sample", {})
        if fi:
            st.caption(
                f"Food.com catalogue after activity filtering: {fi.get('users', 0):,} users · "
                f"{fi.get('recipes', 0):,} recipes · {fi.get('ratings', 0):,} ratings."
            )
