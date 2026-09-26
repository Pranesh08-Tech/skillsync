"""
app.py
------
SkillSync – AI-powered internal code contributor and skill-graph matcher.
Streamlit front-end with five tabs:
  1. Index Contributors  – load GitHub or sample contributor profiles
  2. Search by Skill     – semantic search over contributor profiles
  3. Browse All          – list all indexed contributor profiles
  4. Match Task          – match a task description to the best developer
                           based on their indexed code snippets
  5. Skill Graph         – interactive network of developers ↔ languages
"""

import json
import os
from pathlib import Path

import streamlit as st

import streamlit.components.v1 as components

from backend.engine import SkillSyncEngine
from backend.github_integration import build_contributor_skill_summary
from backend.seed import SEED_DEVELOPERS
from backend.skill_graph import get_graph_stats, render_skill_graph_html
from backend.vector_store import (
    clear_collection,
    list_all_contributors,
    query_similar_contributors,
    upsert_contributors,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SkillSync",
    page_icon="🔗",
    layout="wide",
)

SAMPLE_DATA_PATH = Path("data/sample_contributors.json")

# ---------------------------------------------------------------------------
# Shared engine (cached so the model loads only once per session)
# ---------------------------------------------------------------------------


@st.cache_resource
def get_engine() -> SkillSyncEngine:
    return SkillSyncEngine()


# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------

if "indexed" not in st.session_state:
    st.session_state["indexed"] = False

if "code_indexed" not in st.session_state:
    st.session_state["code_indexed"] = False

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("⚙️ SkillSync Settings")
data_source = st.sidebar.radio(
    "Contributor data source",
    ["Sample dataset", "Live GitHub repo"],
    index=0,
)

if data_source == "Live GitHub repo":
    gh_owner = st.sidebar.text_input("GitHub owner / org", value="")
    gh_repo = st.sidebar.text_input("Repository name", value="")
    gh_token = st.sidebar.text_input(
        "GitHub token (optional, avoids rate limits)",
        type="password",
        value=os.getenv("GITHUB_TOKEN", ""),
    )
    if gh_token:
        os.environ["GITHUB_TOKEN"] = gh_token

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear contributor index"):
    clear_collection()
    st.session_state["indexed"] = False
    st.sidebar.success("Contributor index cleared.")

if st.sidebar.button("🗑️ Clear code index"):
    get_engine().clear()
    st.session_state["code_indexed"] = False
    st.sidebar.success("Code-snippet index cleared.")

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------

st.title("🔗 SkillSync")
st.caption("AI-powered contributor & skill-graph matcher for enterprise repos")

tab_index, tab_search, tab_browse, tab_match, tab_graph = st.tabs(
    ["📥 Index Contributors", "🔍 Search by Skill", "👥 Browse All", "🎯 Match Task", "🕸️ Skill Graph"]
)

# ---------------------------------------------------------------------------
# Tab 1 – Index contributors (profile-level)
# ---------------------------------------------------------------------------

with tab_index:
    st.subheader("Load & index contributor profiles")

    if data_source == "Sample dataset":
        st.info(f"Using bundled sample data from `{SAMPLE_DATA_PATH}`.")
        if st.button("Index sample contributors"):
            with st.spinner("Indexing …"):
                contributors = json.loads(SAMPLE_DATA_PATH.read_text())
                upsert_contributors(contributors)
                st.session_state["indexed"] = True
            st.success(f"Indexed {len(contributors)} contributors.")
    else:
        if not (gh_owner and gh_repo):
            st.warning("Enter a GitHub owner and repository name in the sidebar.")
        else:
            st.info(f"Will fetch contributors from **{gh_owner}/{gh_repo}**.")
            if st.button("Fetch & index from GitHub"):
                with st.spinner("Fetching from GitHub API …"):
                    try:
                        summaries = build_contributor_skill_summary(gh_owner, gh_repo)
                        upsert_contributors(summaries)
                        st.session_state["indexed"] = True
                        st.success(f"Indexed {len(summaries)} contributors.")
                    except Exception as exc:
                        st.error(f"GitHub API error: {exc}")

    if st.session_state["indexed"]:
        st.caption("✅ Index is populated — head to **Search by Skill** or **Browse All**.")

# ---------------------------------------------------------------------------
# Tab 2 – Search by skill (profile-level)
# ---------------------------------------------------------------------------

with tab_search:
    st.subheader("Find contributors by skill")
    query = st.text_input(
        "Describe the skill or technology you need",
        placeholder="e.g. machine learning with Python and Jupyter",
    )
    top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)

    if st.button("Search", disabled=not query):
        if not st.session_state["indexed"]:
            st.warning("Index is empty — please index contributors first.")
        else:
            with st.spinner("Searching …"):
                results = query_similar_contributors(query, n_results=top_k)

            if not results:
                st.info("No results found.")
            else:
                for r in results:
                    score_pct = f"{r['score'] * 100:.1f}%"
                    with st.container():
                        col_a, col_b = st.columns([1, 6])
                        with col_a:
                            st.metric("Match", score_pct)
                        with col_b:
                            st.markdown(
                                f"**[{r['login']}]({r['html_url']})** — "
                                f"{r['contributions']} contributions"
                            )
                            st.caption(
                                "Languages: "
                                + (", ".join(r["languages"]) if r["languages"] else "—")
                            )
                        st.divider()

# ---------------------------------------------------------------------------
# Tab 3 – Browse all (profile-level)
# ---------------------------------------------------------------------------

with tab_browse:
    st.subheader("All indexed contributors")
    if st.button("Refresh list"):
        pass  # triggers a rerun naturally

    all_contribs = list_all_contributors()
    if not all_contribs:
        st.info("No contributors indexed yet.")
    else:
        st.caption(f"{len(all_contribs)} contributors in the index.")
        for c in sorted(all_contribs, key=lambda x: x["contributions"], reverse=True):
            with st.expander(
                f"**{c['login']}** — {c['contributions']} contributions"
            ):
                st.write(
                    "**Languages:**",
                    ", ".join(c["languages"]) if c["languages"] else "—",
                )
                if c["html_url"]:
                    st.write(f"**GitHub:** [{c['html_url']}]({c['html_url']})")
                st.caption(c["document"])

# ---------------------------------------------------------------------------
# Tab 4 – Match task to developer (code-snippet level)
# ---------------------------------------------------------------------------

with tab_match:
    st.subheader("Match a task to the best-fit developer")
    st.caption(
        "This tab uses code-snippet embeddings (indexed via the engine) "
        "rather than high-level contributor profiles."
    )

    # --- Seed code snippets ---
    with st.expander("📦 Seed code-snippet index", expanded=not st.session_state["code_indexed"]):
        st.markdown(
            f"Load **{len(SEED_DEVELOPERS)} sample code snippets** from internal mock data."
        )
        if st.button("Seed sample developer code"):
            engine = get_engine()
            with st.spinner("Seeding …"):
                for entry in SEED_DEVELOPERS:
                    engine.add_developer_code(
                        dev_name=entry["dev_name"],
                        code_snippet=entry["code_snippet"],
                        repo_name=entry["repo_name"],
                    )
            st.session_state["code_indexed"] = True
            st.success(
                f"Seeded {len(SEED_DEVELOPERS)} snippets. "
                f"Total in index: {get_engine().count()}"
            )

    st.divider()

    # --- Task matching ---
    task = st.text_area(
        "Describe your task",
        placeholder=(
            "e.g. We need to add OAuth2 token support to the users table "
            "and migrate existing records."
        ),
        height=100,
    )
    n_devs = st.slider("Developers to suggest", min_value=1, max_value=10, value=3)

    if st.button("Find best-fit developer(s)", disabled=not task):
        engine = get_engine()
        if engine.count() == 0:
            st.warning("Code-snippet index is empty — seed it first using the panel above.")
        else:
            with st.spinner("Matching …"):
                matches = engine.query_developers(task, n_results=n_devs)

            if not matches:
                st.info("No matches found.")
            else:
                for rank, m in enumerate(matches, start=1):
                    score_pct = f"{m['score'] * 100:.1f}%"
                    with st.container():
                        col_rank, col_detail = st.columns([1, 7])
                        with col_rank:
                            st.metric(f"#{rank}", score_pct)
                        with col_detail:
                            st.markdown(
                                f"**{m['developer']}** — repo: `{m['repo']}`"
                            )
                            st.code(m["code_snippet"], language="python")
                        st.divider()

# ---------------------------------------------------------------------------
# Tab 5 – Skill Graph
# ---------------------------------------------------------------------------

with tab_graph:
    st.subheader("Skill Graph")
    st.caption(
        "Interactive network of contributors (🔵 blue dots) connected to the "
        "languages/skills they use (🟣 purple diamonds). "
        "Node size scales with contribution count / skill popularity."
    )

    all_for_graph = list_all_contributors()

    if not all_for_graph:
        st.info(
            "No contributor profiles indexed yet. "
            "Go to **📥 Index Contributors** and load data first."
        )
    else:
        # --- Summary metrics ---
        stats = get_graph_stats(all_for_graph)
        c1, c2, c3 = st.columns(3)
        c1.metric("Developers", stats["num_developers"])
        c2.metric("Unique Skills", stats["num_skills"])
        c3.metric("Connections", stats["num_edges"])

        if stats["top_skills"]:
            st.markdown("**Top skills by number of contributors:**")
            for entry in stats["top_skills"]:
                st.write(f"- **{entry['skill']}** — {entry['count']} developer(s)")

        st.divider()

        # --- Interactive network ---
        graph_height = st.select_slider(
            "Graph height", options=["400px", "500px", "600px", "700px", "800px"], value="600px"
        )

        with st.spinner("Rendering skill graph …"):
            html = render_skill_graph_html(all_for_graph, height=graph_height)

        components.html(html, height=int(graph_height.replace("px", "")) + 20, scrolling=False)
