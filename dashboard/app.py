"""CouchHire Dashboard — Streamlit application for tracking and analytics.

Step 26: Interactive dashboard with four tabs:
  1. Tracker   — live table of all applications with filtering and outcome labeling
  2. Analytics — interactive Plotly charts and summary stats
  3. Retrain   — trigger NLP model retraining, view label stats
  4. Settings  — view/edit config.yaml and form_answers.json

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so relative imports work when
# Streamlit is launched from the project root directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio

# ---------------------------------------------------------------------------
# Plotly theme — transparent backgrounds for light/dark mode compatibility
# ---------------------------------------------------------------------------
_transparent_layout = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="rgba(150,150,150,1)"),
    xaxis=dict(gridcolor="rgba(128,128,128,0.2)", zerolinecolor="rgba(128,128,128,0.2)"),
    yaxis=dict(gridcolor="rgba(128,128,128,0.2)", zerolinecolor="rgba(128,128,128,0.2)"),
)
pio.templates["couchhire"] = pio.templates["plotly"]
pio.templates["couchhire"].layout.update(_transparent_layout)
pio.templates.default = "couchhire"

import plotly.graph_objects as go

from dashboard.helpers import (
    STATUS_OPTIONS,
    OUTCOME_OPTIONS,
    ROUTE_OPTIONS,
    OUTCOME_COLORS,
    ROUTE_COLORS,
    STATUS_EMOJI,
    OUTCOME_EMOJI,
    applications_to_dataframe,
    compute_summary_stats,
    format_score_badge,
    format_status,
    format_outcome,
    load_form_answers,
    save_form_answers,
    load_settings,
    save_settings,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_CONFIG_YAML_PATH = _PROJECT_ROOT / "config.yaml"
_FORM_ANSWERS_PATH = _PROJECT_ROOT / "cv" / "uploads" / "form_answers.json"

# Try to get the form answers path from config (if importable)
try:
    from config import FORM_ANSWERS_PATH as _CFG_FORM_ANSWERS_PATH
    _FORM_ANSWERS_PATH = _CFG_FORM_ANSWERS_PATH
except Exception:
    pass

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CouchHire Dashboard",
    page_icon="dashboard/CouchHire.jpeg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Stat card styling — works in both light and dark mode */
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.1);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 10px;
        padding: 15px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.85rem;
        opacity: 0.75;
    }
    /* Table row hover — subtle in both modes */
    .stDataFrame tbody tr:hover {
        background-color: rgba(128, 128, 128, 0.15) !important;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        font-weight: 600;
    }
    /* Compact expander */
    .streamlit-expanderHeader {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30)
def _load_applications() -> list[dict]:
    """Fetch all applications from Supabase with 30s cache."""
    try:
        from db.supabase_client import get_all_applications
        return get_all_applications(limit=500)
    except Exception as exc:
        logger.error("Failed to load applications: %s", exc)
        st.error(f"⚠️ Could not load applications from Supabase: {exc}")
        return []


@st.cache_data(ttl=30)
def _load_labeled_outcomes() -> list[dict]:
    """Fetch labeled outcomes for retrain stats."""
    try:
        from db.supabase_client import get_labeled_outcomes
        return get_labeled_outcomes()
    except Exception as exc:
        logger.error("Failed to load labeled outcomes: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("dashboard/CouchHire.jpeg", width=80)
    st.title("CouchHire")
    st.caption("Agentic Job Application Pipeline")
    st.divider()

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown("**Quick Stats**")

    raw_apps = _load_applications()
    df_all = applications_to_dataframe(raw_apps)
    stats = compute_summary_stats(df_all)

    st.metric("Total Applications", stats["total"])
    st.metric("Avg Match Score", f"{stats['avg_score']}%")
    st.metric("Interview Rate", f"{stats['interview_rate']}%")
    st.metric("Applied", stats["applied_count"])

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab_tracker, tab_analytics, tab_retrain, tab_settings = st.tabs([
    "📋 Tracker", "📊 Analytics", "🧠 Retrain", "⚙️ Settings"
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: TRACKER
# ═══════════════════════════════════════════════════════════════════════════
with tab_tracker:
    st.header("Application Tracker")

    if df_all.empty:
        st.info("No applications found. Run the pipeline to get started!")
    else:
        # --- Filters ---
        filter_cols = st.columns([1, 1, 1, 2])

        with filter_cols[0]:
            filter_status = st.multiselect(
                "Status",
                options=STATUS_OPTIONS,
                default=[],
                placeholder="All statuses",
            )
        with filter_cols[1]:
            filter_outcome = st.multiselect(
                "Outcome",
                options=OUTCOME_OPTIONS,
                default=[],
                placeholder="All outcomes",
            )
        with filter_cols[2]:
            filter_route = st.multiselect(
                "Route",
                options=ROUTE_OPTIONS,
                default=[],
                placeholder="All routes",
            )
        with filter_cols[3]:
            if "date" in df_all.columns and not df_all["date"].isna().all():
                min_date = df_all["date"].min()
                max_date = df_all["date"].max()
                date_range = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                )
            else:
                date_range = None

        # --- Apply filters ---
        df_filtered = df_all.copy()

        if filter_status:
            df_filtered = df_filtered[df_filtered["status"].isin(filter_status)]
        if filter_outcome:
            df_filtered = df_filtered[df_filtered["outcome"].isin(filter_outcome)]
        if filter_route:
            df_filtered = df_filtered[df_filtered["route"].isin(filter_route)]
        if date_range and len(date_range) == 2 and "date" in df_filtered.columns:
            df_filtered = df_filtered[
                (df_filtered["date"] >= date_range[0])
                & (df_filtered["date"] <= date_range[1])
            ]

        st.caption(f"Showing {len(df_filtered)} of {len(df_all)} applications")

        # --- Display columns ---
        display_cols = [
            "company", "role", "match_score", "route", "status", "outcome", "date",
        ]
        available_display_cols = [c for c in display_cols if c in df_filtered.columns]

        if "draft_url" in df_filtered.columns:
            available_display_cols.append("draft_url")

        # Sort
        sort_col = st.selectbox(
            "Sort by",
            options=available_display_cols,
            index=available_display_cols.index("date") if "date" in available_display_cols else 0,
            key="tracker_sort",
        )
        sort_asc = st.checkbox("Ascending", value=False, key="tracker_sort_asc")
        df_sorted = df_filtered.sort_values(
            by=sort_col,
            ascending=sort_asc,
            na_position="last",
        )

        # --- Render table with expandable rows ---
        for idx, row in df_sorted.iterrows():
            company = row.get("company", "—")
            role = row.get("role", "—")
            score = row.get("match_score")
            status = row.get("status", "—")
            outcome_val = row.get("outcome", "—")
            route = row.get("route", "—")
            date_val = row.get("date", "—")
            draft_url = row.get("draft_url", "")
            app_id = row.get("id", "")

            score_badge = format_score_badge(score)
            status_display = format_status(status)
            outcome_display = format_outcome(outcome_val)

            # Row header
            header = f"**{company}** — {role} | {score_badge} | {status_display} | {outcome_display} | {date_val}"

            with st.expander(header, expanded=False):
                detail_cols = st.columns([2, 2, 1])

                with detail_cols[0]:
                    st.markdown(f"**Company:** {company}")
                    st.markdown(f"**Role:** {role}")
                    st.markdown(f"**Match Score:** {score_badge}")
                    st.markdown(f"**Route:** {route}")
                    st.markdown(f"**Status:** {status_display}")
                    st.markdown(f"**Date:** {date_val}")
                    if draft_url:
                        st.markdown(f"[📧 Open Draft]({draft_url})")

                with detail_cols[1]:
                    # Full details
                    jd_text = row.get("jd_text", "")
                    resume_content = row.get("resume_content", "")
                    cover_letter = row.get("cover_letter", "")
                    email_body = row.get("email_body", "")
                    email_subject = row.get("email_subject", "")

                    if email_subject:
                        st.markdown(f"**Email Subject:** {email_subject}")
                    if email_body:
                        with st.container():
                            st.markdown("**Email Body:**")
                            st.text_area(
                                "email_body",
                                value=email_body,
                                height=120,
                                disabled=True,
                                key=f"email_{app_id}",
                                label_visibility="collapsed",
                            )
                    if cover_letter:
                        with st.container():
                            st.markdown("**Cover Letter:**")
                            st.text_area(
                                "cover_letter",
                                value=cover_letter,
                                height=120,
                                disabled=True,
                                key=f"cl_{app_id}",
                                label_visibility="collapsed",
                            )

                with detail_cols[2]:
                    # Inline outcome labeling
                    st.markdown("**Set Outcome:**")
                    current_outcome_idx = 0
                    outcome_choices = ["— (none)"] + OUTCOME_OPTIONS
                    if outcome_val in OUTCOME_OPTIONS:
                        current_outcome_idx = outcome_choices.index(outcome_val)

                    new_outcome = st.selectbox(
                        "Outcome",
                        options=outcome_choices,
                        index=current_outcome_idx,
                        key=f"outcome_select_{app_id}",
                        label_visibility="collapsed",
                    )

                    if st.button("💾 Save", key=f"save_outcome_{app_id}"):
                        if new_outcome != "— (none)" and app_id:
                            try:
                                from db.supabase_client import update_outcome
                                update_outcome(str(app_id), new_outcome)
                                st.success(f"Outcome set to **{new_outcome}**")
                                st.cache_data.clear()
                            except Exception as exc:
                                st.error(f"Failed to update: {exc}")
                        elif new_outcome == "— (none)":
                            st.warning("Select an outcome first")

                # JD and resume in a sub-expander
                if jd_text:
                    with st.expander("📄 Job Description", expanded=False):
                        st.text_area(
                            "jd",
                            value=jd_text,
                            height=200,
                            disabled=True,
                            key=f"jd_{app_id}",
                            label_visibility="collapsed",
                        )
                if resume_content:
                    with st.expander("📝 Resume Content", expanded=False):
                        st.text_area(
                            "resume",
                            value=resume_content,
                            height=200,
                            disabled=True,
                            key=f"resume_{app_id}",
                            label_visibility="collapsed",
                        )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════
with tab_analytics:
    st.header("Analytics")

    if df_all.empty:
        st.info("No application data to analyse yet. Run the pipeline first!")
    else:
        # --- Summary stat cards ---
        card_cols = st.columns(6)
        card_data = [
            ("📊 Total Apps", stats["total"], None),
            ("🎯 Avg Score", f"{stats['avg_score']}%", None),
            ("🎤 Interview Rate", f"{stats['interview_rate']}%", None),
            ("🎉 Offer Rate", f"{stats['offer_rate']}%", None),
            ("✅ Applied", stats["applied_count"], None),
            ("🏷️ Labeled", stats.get("labeled_count", 0), None),
        ]
        for col, (label, value, delta) in zip(card_cols, card_data):
            col.metric(label=label, value=value, delta=delta)

        st.divider()

        # --- Row 1: Applications over time + Match score distribution ---
        chart_row1 = st.columns(2)

        with chart_row1[0]:
            st.subheader("Applications Over Time")
            if "date" in df_all.columns and not df_all["date"].isna().all():
                apps_by_date = (
                    df_all.groupby("date")
                    .size()
                    .reset_index(name="count")
                    .sort_values("date")
                )
                fig_timeline = px.bar(
                    apps_by_date,
                    x="date",
                    y="count",
                    labels={"date": "Date", "count": "Applications"},
                    color_discrete_sequence=["#4CAF50"],
                )
                fig_timeline.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Applications",
                    showlegend=False,
                    height=350,
                    margin=dict(l=20, r=20, t=30, b=20),
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
            else:
                st.caption("No date data available")

        with chart_row1[1]:
            st.subheader("Match Score Distribution")
            if "match_score" in df_all.columns and df_all["match_score"].notna().any():
                fig_hist = px.histogram(
                    df_all.dropna(subset=["match_score"]),
                    x="match_score",
                    nbins=20,
                    labels={"match_score": "Match Score"},
                    color_discrete_sequence=["#2196F3"],
                )
                fig_hist.update_layout(
                    xaxis_title="Match Score",
                    yaxis_title="Count",
                    showlegend=False,
                    height=350,
                    margin=dict(l=20, r=20, t=30, b=20),
                )
                # Add threshold line
                try:
                    from config import MATCH_THRESHOLD
                    fig_hist.add_vline(
                        x=MATCH_THRESHOLD,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Threshold ({MATCH_THRESHOLD})",
                    )
                except Exception:
                    pass
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.caption("No score data available")

        # --- Row 2: Outcome breakdown + Route split ---
        chart_row2 = st.columns(2)

        with chart_row2[0]:
            st.subheader("Outcome Breakdown")
            if "outcome" in df_all.columns:
                outcome_counts = (
                    df_all[df_all["outcome"].isin(OUTCOME_OPTIONS)]
                    .groupby("outcome")
                    .size()
                    .reset_index(name="count")
                )
                if not outcome_counts.empty:
                    colors = [
                        OUTCOME_COLORS.get(o, "#999")
                        for o in outcome_counts["outcome"]
                    ]
                    fig_outcome = go.Figure(
                        data=[go.Pie(
                            labels=outcome_counts["outcome"],
                            values=outcome_counts["count"],
                            hole=0.45,
                            marker=dict(colors=colors),
                            textinfo="label+percent+value",
                            textposition="auto",
                        )]
                    )
                    fig_outcome.update_layout(
                        height=350,
                        margin=dict(l=20, r=20, t=30, b=20),
                        showlegend=True,
                    )
                    st.plotly_chart(fig_outcome, use_container_width=True)
                else:
                    st.caption("No labeled outcomes yet — use /outcome in Telegram or the Tracker tab")
            else:
                st.caption("No outcome data available")

        with chart_row2[1]:
            st.subheader("Route Split")
            if "route" in df_all.columns:
                route_counts = (
                    df_all[df_all["route"].isin(ROUTE_OPTIONS)]
                    .groupby("route")
                    .size()
                    .reset_index(name="count")
                )
                if not route_counts.empty:
                    colors = [
                        ROUTE_COLORS.get(r, "#999")
                        for r in route_counts["route"]
                    ]
                    fig_route = go.Figure(
                        data=[go.Pie(
                            labels=route_counts["route"],
                            values=route_counts["count"],
                            hole=0.45,
                            marker=dict(colors=colors),
                            textinfo="label+percent+value",
                            textposition="auto",
                        )]
                    )
                    fig_route.update_layout(
                        height=350,
                        margin=dict(l=20, r=20, t=30, b=20),
                        showlegend=True,
                    )
                    st.plotly_chart(fig_route, use_container_width=True)
                else:
                    st.caption("No route data yet")
            else:
                st.caption("No route data available")

        # --- Row 3: Score vs Outcome correlation ---
        st.divider()
        st.subheader("Score vs Outcome")
        if (
            "match_score" in df_all.columns
            and "outcome" in df_all.columns
            and df_all["outcome"].isin(OUTCOME_OPTIONS).any()
        ):
            df_labeled = df_all[
                df_all["outcome"].isin(OUTCOME_OPTIONS)
                & df_all["match_score"].notna()
            ].copy()

            if not df_labeled.empty:
                # Grouped bar: average score per outcome
                avg_by_outcome = (
                    df_labeled.groupby("outcome")["match_score"]
                    .agg(["mean", "count"])
                    .reset_index()
                    .rename(columns={"mean": "avg_score", "count": "num_apps"})
                )
                # Order by outcome quality
                outcome_order = ["offer", "interview", "no_response", "rejected", "withdrawn"]
                avg_by_outcome["outcome"] = pd.Categorical(
                    avg_by_outcome["outcome"],
                    categories=outcome_order,
                    ordered=True,
                )
                avg_by_outcome = avg_by_outcome.sort_values("outcome")

                colors = [
                    OUTCOME_COLORS.get(o, "#999")
                    for o in avg_by_outcome["outcome"]
                ]

                fig_corr = go.Figure(
                    data=[go.Bar(
                        x=avg_by_outcome["outcome"],
                        y=avg_by_outcome["avg_score"],
                        text=[
                            f"{s:.1f}% (n={n})"
                            for s, n in zip(
                                avg_by_outcome["avg_score"],
                                avg_by_outcome["num_apps"],
                            )
                        ],
                        textposition="auto",
                        marker_color=colors,
                    )]
                )
                fig_corr.update_layout(
                    xaxis_title="Outcome",
                    yaxis_title="Average Match Score",
                    height=350,
                    margin=dict(l=20, r=20, t=30, b=20),
                    showlegend=False,
                )
                st.plotly_chart(fig_corr, use_container_width=True)

                # Also show scatter
                with st.expander("📈 Scatter: Individual Scores by Outcome"):
                    fig_scatter = px.strip(
                        df_labeled,
                        x="outcome",
                        y="match_score",
                        color="outcome",
                        color_discrete_map=OUTCOME_COLORS,
                        labels={"match_score": "Match Score", "outcome": "Outcome"},
                        category_orders={"outcome": outcome_order},
                    )
                    fig_scatter.update_layout(
                        height=300,
                        margin=dict(l=20, r=20, t=30, b=20),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.caption("Need labeled outcomes with match scores for this chart")
        else:
            st.caption("Need labeled outcomes with match scores for this chart")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: RETRAIN
# ═══════════════════════════════════════════════════════════════════════════
with tab_retrain:
    st.header("Model Retraining")
    st.caption(
        "Fine-tune the ATS match scorer using real application outcomes. "
        "The model learns from your labeled results (interview, rejected, etc.)."
    )

    # --- Current label stats ---
    labeled_data = _load_labeled_outcomes()
    labeled_df = applications_to_dataframe(labeled_data)

    retrain_cols = st.columns([2, 1])

    with retrain_cols[0]:
        st.subheader("Label Statistics")

        if labeled_data:
            from collections import Counter
            outcome_counter = Counter(row.get("outcome", "unknown") for row in labeled_data)
            total_labeled = sum(outcome_counter.values())

            label_cols = st.columns(len(OUTCOME_OPTIONS))
            for col, outcome in zip(label_cols, OUTCOME_OPTIONS):
                count = outcome_counter.get(outcome, 0)
                emoji = OUTCOME_EMOJI.get(outcome, "")
                col.metric(
                    label=f"{emoji} {outcome.title()}",
                    value=count,
                )

            st.divider()

            # Class balance visualization
            st.markdown("**Class Balance:**")
            balance_data = pd.DataFrame([
                {"outcome": k, "count": v}
                for k, v in outcome_counter.items()
                if k in OUTCOME_OPTIONS
            ])
            if not balance_data.empty:
                colors = [
                    OUTCOME_COLORS.get(o, "#999")
                    for o in balance_data["outcome"]
                ]
                fig_balance = px.bar(
                    balance_data,
                    x="outcome",
                    y="count",
                    color="outcome",
                    color_discrete_map=OUTCOME_COLORS,
                    labels={"outcome": "Outcome", "count": "Count"},
                )
                fig_balance.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=20, b=20),
                    showlegend=False,
                )
                st.plotly_chart(fig_balance, use_container_width=True)

            st.info(f"📊 **{total_labeled}** total labeled outcomes")
        else:
            st.warning("No labeled outcomes yet. Label applications in the Tracker tab or via Telegram /outcome.")

    with retrain_cols[1]:
        st.subheader("Retrain Controls")

        # Auto-retrain indicator
        try:
            from nlp.retrain import should_retrain
            auto_ready = should_retrain()
            if auto_ready:
                st.success("✅ Auto-retrain threshold met — ready to retrain!")
            else:
                st.info("⏳ Auto-retrain threshold not yet met")
        except Exception as exc:
            st.warning(f"Could not check auto-retrain status: {exc}")

        # Show config values
        try:
            from config import MIN_RETRAIN_LABELS, RETRAIN_EVERY
            st.markdown(f"**Min labels required:** {MIN_RETRAIN_LABELS}")
            st.markdown(f"**Retrain every:** {RETRAIN_EVERY} labels")
        except Exception:
            pass

        st.divider()

        # Retrain button
        st.markdown("**Manual Retrain:**")
        force_retrain = st.checkbox("Force retrain (skip minimum check)", value=False)

        if st.button("🧠 Retrain Now", type="primary", use_container_width=True):
            with st.spinner("Training model... this may take a minute."):
                try:
                    from nlp.retrain import retrain
                    result = retrain(force=force_retrain)

                    if result["status"] == "success":
                        st.success("✅ Model retrained successfully!")
                        st.balloons()
                    elif result["status"] == "skipped":
                        st.warning(f"⏭️ Skipped: {result['reason']}")
                    else:
                        st.error(f"❌ Failed: {result['reason']}")

                    # Display result details
                    st.divider()
                    st.markdown("**Retrain Results:**")
                    result_cols = st.columns(2)
                    with result_cols[0]:
                        st.markdown(f"- **Status:** {result['status']}")
                        st.markdown(f"- **Labels used:** {result['num_labels']}")
                        st.markdown(f"- **Positive:** {result['num_positive']}")
                        st.markdown(f"- **Negative:** {result['num_negative']}")
                    with result_cols[1]:
                        st.markdown(f"- **Excluded:** {result['num_excluded']}")
                        st.markdown(f"- **Epochs:** {result['epochs']}")
                        if result.get("model_path"):
                            st.markdown(f"- **Model path:** `{result['model_path']}`")
                        if result.get("reason"):
                            st.markdown(f"- **Reason:** {result['reason']}")

                except Exception as exc:
                    st.error(f"❌ Retrain error: {exc}")
                    logger.error("Retrain failed: %s", exc, exc_info=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: SETTINGS
# ═══════════════════════════════════════════════════════════════════════════
with tab_settings:
    st.header("Settings")

    settings_tab1, settings_tab2 = st.tabs(["🔧 Configuration", "📝 Form Answers"])

    # --- Settings sub-tab 1: config.yaml ---
    with settings_tab1:
        st.subheader("Dashboard Configuration")
        st.caption(
            "Edit dashboard-level settings. These are stored in `config.yaml` "
            "and do NOT modify your `.env` file or secrets."
        )

        settings = load_settings(_CONFIG_YAML_PATH)

        with st.form("settings_form"):
            edited_settings: dict = {}

            settings_cols = st.columns(2)

            with settings_cols[0]:
                st.markdown("**Matching & Scoring**")
                edited_settings["match_threshold"] = st.number_input(
                    "Match Threshold (%)",
                    min_value=0,
                    max_value=100,
                    value=int(settings.get("match_threshold", 60)),
                    step=5,
                    help="Minimum match score to proceed with resume generation",
                )
                edited_settings["min_match_score"] = st.number_input(
                    "Min Match Score for Search (%)",
                    min_value=0,
                    max_value=100,
                    value=int(settings.get("min_match_score", 60)),
                    step=5,
                    help="Minimum match % for job search results",
                )
                edited_settings["max_search_results"] = st.number_input(
                    "Max Search Results",
                    min_value=1,
                    max_value=100,
                    value=int(settings.get("max_search_results", 10)),
                    step=1,
                )

                st.markdown("**Retraining**")
                edited_settings["min_retrain_labels"] = st.number_input(
                    "Min Retrain Labels",
                    min_value=1,
                    max_value=100,
                    value=int(settings.get("min_retrain_labels", 10)),
                    step=1,
                    help="Minimum labeled outcomes before retraining is allowed",
                )
                edited_settings["retrain_every"] = st.number_input(
                    "Retrain Every N Labels",
                    min_value=0,
                    max_value=100,
                    value=int(settings.get("retrain_every", 10)),
                    step=1,
                    help="Auto-retrain every N labels (0 = manual only)",
                )

            with settings_cols[1]:
                st.markdown("**Job Search**")
                edited_settings["jobspy_sites"] = st.text_input(
                    "Job Sites (comma-separated)",
                    value=settings.get("jobspy_sites", "indeed,linkedin,google"),
                    help="Job boards to search: indeed, linkedin, google, glassdoor, zip_recruiter",
                )
                edited_settings["jobspy_country"] = st.text_input(
                    "Country",
                    value=settings.get("jobspy_country", "USA"),
                )
                edited_settings["jobspy_hours_old"] = st.number_input(
                    "Max Job Age (hours)",
                    min_value=1,
                    max_value=720,
                    value=int(settings.get("jobspy_hours_old", 72)),
                    step=12,
                )

                st.markdown("**Browser**")
                edited_settings["browser_headless"] = st.checkbox(
                    "Headless Browser",
                    value=bool(settings.get("browser_headless", False)),
                    help="Run Playwright browser in headless mode",
                )

            submitted = st.form_submit_button(
                "💾 Save Settings",
                type="primary",
                use_container_width=True,
            )

            if submitted:
                if save_settings(_CONFIG_YAML_PATH, edited_settings):
                    st.success("✅ Settings saved to `config.yaml`")
                else:
                    st.error("❌ Failed to save settings")

    # --- Settings sub-tab 2: form_answers.json ---
    with settings_tab2:
        st.subheader("Form Answer Memory")
        st.caption(
            "These are saved answers used by the browser agent when filling ATS forms. "
            "Edit values below and save to update the answer memory."
        )

        form_answers = load_form_answers(_FORM_ANSWERS_PATH)

        if form_answers:
            import json

            # Display as editable JSON
            edited_json = st.text_area(
                "form_answers.json",
                value=json.dumps(form_answers, indent=2, ensure_ascii=False),
                height=400,
                key="form_answers_editor",
                label_visibility="collapsed",
            )

            fa_cols = st.columns([1, 4])
            with fa_cols[0]:
                if st.button("💾 Save Changes", key="save_form_answers"):
                    try:
                        parsed = json.loads(edited_json)
                        if save_form_answers(_FORM_ANSWERS_PATH, parsed):
                            st.success("✅ Form answers saved!")
                        else:
                            st.error("❌ Failed to write file")
                    except json.JSONDecodeError as exc:
                        st.error(f"❌ Invalid JSON: {exc}")

            # Show as table for quick reference
            with st.expander("📊 View as Table"):
                if isinstance(form_answers, dict):
                    flat_items = []
                    for key, value in form_answers.items():
                        if isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                flat_items.append({
                                    "Category": key,
                                    "Field": sub_key,
                                    "Value": str(sub_value),
                                })
                        else:
                            flat_items.append({
                                "Category": "—",
                                "Field": key,
                                "Value": str(value),
                            })
                    if flat_items:
                        st.dataframe(
                            pd.DataFrame(flat_items),
                            use_container_width=True,
                            hide_index=True,
                        )
        else:
            st.info(
                "No form answers file found. The browser agent will create one "
                "automatically when it first fills an ATS form."
            )
            st.caption(f"Expected path: `{_FORM_ANSWERS_PATH}`")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption("🛋️ CouchHire Dashboard — Made with ❤️ by Satish")
