"""
demo_app.py — FNOL Triage Workbench
Streamlit three-panel decision support interface.

Run with:
    streamlit run demo_app.py

Requires inference API running at localhost:8000:
    uvicorn api:app --port 8000
"""

import os
import streamlit as st
import requests
import json
import re
import html as html_lib

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="FNOL Triage Workbench",
    layout="wide",
    page_icon="🔍",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { padding-top: 1rem; }
  .triage-card {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
  }
  .conf-high { color: #2e7d32; font-weight: bold; }
  .conf-medium { color: #e65100; font-weight: bold; }
  .conf-low { color: #c62828; font-weight: bold; }
  .routing-llu { background: #b71c1c; color: white; padding: 4px 12px; border-radius: 4px; font-weight: bold; }
  .routing-lmu { background: #e65100; color: white; padding: 4px 12px; border-radius: 4px; font-weight: bold; }
  .routing-stp { background: #2e7d32; color: white; padding: 4px 12px; border-radius: 4px; font-weight: bold; }
  .routing-siu { background: #4a148c; color: white; padding: 4px 12px; border-radius: 4px; font-weight: bold; }
  .signal-badge {
    display: inline-block;
    background: #e8eaf6;
    border: 1px solid #9fa8da;
    color: #283593;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 13px;
    margin: 3px;
  }
  .review-banner {
    background: #fff3e0;
    border: 2px solid #e65100;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 15px;
    font-weight: bold;
    color: #bf360c;
  }
  .critical-banner {
    background: #ffebee;
    border: 2px solid #c62828;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 15px;
    font-weight: bold;
    color: #b71c1c;
  }
</style>
""", unsafe_allow_html=True)

# ── Sample FNOLs ──────────────────────────────────────────────────────────────
@st.cache_data
def get_sample_fnols():
    try:
        r = requests.get(f"{API_URL}/sample-fnols", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []

# ── Helpers ───────────────────────────────────────────────────────────────────
def highlight_narrative(narrative: str, highlights: list) -> str:
    """Render narrative HTML with colored highlight spans."""
    if not highlights:
        return html_lib.escape(narrative).replace("\n", "<br>")

    # Sort by start position, reverse order for safe insertion
    sorted_highlights = sorted(highlights, key=lambda h: h["start"], reverse=True)
    chars = list(narrative)

    for h in sorted_highlights:
        start, end = h["start"], h["end"]
        color = h.get("color", "#FFD54F")
        explanation = html_lib.escape(h["explanation"])
        span_open = (f'<mark style="background:{color}22;border-bottom:2px solid {color};'
                     f'cursor:help" title="{explanation}">')
        span_close = "</mark>"
        chars[start:end] = list(span_open + "".join(chars[start:end]) + span_close)

    result = "".join(chars)
    return html_lib.escape(result).replace(span_open, span_open).replace(span_close, span_close).replace("\n", "<br>")


def routing_badge(routing: str) -> str:
    css_class = f"routing-{routing.lower()}"
    return f'<span class="{css_class}">{routing}</span>'


def confidence_badge(conf: str) -> str:
    return f'<span class="conf-{conf}">{"●" * (3 if conf=="high" else 2 if conf=="medium" else 1)} {conf.upper()}</span>'


# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🔍 FNOL Triage Workbench")
st.caption("Commercial Lines Claims Triage — Fine-Tuned Mistral-7B")

# Samples row
sample_fnols = get_sample_fnols()
if sample_fnols:
    st.markdown("**Quick test cases:**")
    cols = st.columns(len(sample_fnols))
    for col, sample in zip(cols, sample_fnols):
        with col:
            if st.button(sample["label"], key=f"sample_{sample['id']}"):
                st.session_state["narrative_input"] = sample["narrative"]
                st.session_state["channel_input"] = sample["channel"]

st.divider()

# Three panels
left, center, right = st.columns([1.2, 1.3, 1.5])

# ── LEFT PANEL ────────────────────────────────────────────────────────────────
with left:
    st.subheader("📋 FNOL Input")

    narrative = st.text_area(
        "Paste or type the FNOL narrative",
        value=st.session_state.get("narrative_input", ""),
        height=320,
        placeholder="Enter raw FNOL narrative here...",
        key="narrative_text",
    )

    channel = st.selectbox(
        "Submission channel",
        ["broker_email", "portal", "fax_to_email"],
        index=["broker_email", "portal", "fax_to_email"].index(
            st.session_state.get("channel_input", "portal")
        ),
    )

    compare_mode = st.checkbox("🔀 Split-view: compare base vs. fine-tuned model")

    triage_clicked = st.button("🚀 Triage", type="primary", use_container_width=True)

# ── State management ──────────────────────────────────────────────────────────
if triage_clicked and narrative.strip():
    with st.spinner("Running triage..."):
        try:
            response = requests.post(
                f"{API_URL}/triage",
                json={
                    "narrative": narrative,
                    "submission_channel": channel,
                    "compare_base_model": compare_mode,
                },
                timeout=60,
            )
            if response.status_code == 200:
                st.session_state["triage_result"] = response.json()
                st.session_state["triage_narrative"] = narrative
            else:
                st.error(f"API error: {response.status_code} — {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to inference API. Is `uvicorn api:app --port 8000` running?")

result = st.session_state.get("triage_result")
original_narrative = st.session_state.get("triage_narrative", "")

# ── CENTER PANEL ──────────────────────────────────────────────────────────────
with center:
    st.subheader("📊 Model Output")

    if result:
        # Human review banner — most prominent element
        if result.get("human_review_required"):
            severity_val = result.get("severity_tier", {}).get("value", "")
            banner_class = "critical-banner" if severity_val == "C1" else "review-banner"
            reasons = "<br>• ".join(result.get("human_review_reasons", []))
            st.markdown(
                f'<div class="{banner_class}">⚠️ HUMAN REVIEW REQUIRED<br>'
                f'<span style="font-weight:normal;font-size:13px">• {reasons}</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#e8f5e9;border:1px solid #4caf50;border-radius:6px;'
                'padding:10px 14px;color:#2e7d32;font-weight:bold">✅ Auto-route eligible</div>',
                unsafe_allow_html=True,
            )

        st.markdown("")

        # Triage output cards
        dims = [
            ("Coverage Line", result.get("coverage_line", {})),
            ("Cause Code", result.get("cause_code", {})),
            ("Severity Tier", result.get("severity_tier", {})),
            ("Routing Decision", result.get("routing", {})),
        ]

        for label, dim in dims:
            val = dim.get("value", "—")
            conf = dim.get("confidence", "low")

            # Special rendering for routing
            if label == "Routing Decision":
                val_html = routing_badge(val)
            elif label == "Severity Tier" and val == "C1":
                val_html = f'<strong style="color:#c62828;font-size:18px">{val}</strong>'
            else:
                val_html = f"<strong style='font-size:16px'>{val}</strong>"

            st.markdown(
                f'<div class="triage-card">'
                f'<div style="font-size:12px;color:#757575;margin-bottom:4px">{label}</div>'
                f'{val_html} &nbsp; {confidence_badge(conf)}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Domain signals
        signals = result.get("domain_signals_detected", [])
        st.markdown("**Detected domain signals:**")
        if signals:
            badges = " ".join(
                f'<span class="signal-badge">{s.replace("_", " ")}</span>'
                for s in signals
            )
            st.markdown(badges, unsafe_allow_html=True)
        else:
            st.markdown("_No special signals detected_")

        # Base model comparison (split-view)
        if compare_mode and result.get("base_model_output"):
            st.divider()
            st.markdown("**🔀 Base model output (untuned):**")
            base = result["base_model_output"]
            base_routing = base.get("routing", "?")
            base_sev = base.get("severity_tier", "?")
            st.markdown(
                f"Routing: **{base_routing}** | Severity: **{base_sev}** | "
                f"Confidence: **{base.get('confidence', '?'):.0%}**"
            )

        st.caption(f"Processing time: {result.get('processing_time_ms', 0)}ms | "
                   f"FNOL ID: {result.get('fnol_id', '—')}")
    else:
        st.info("Submit a narrative to see triage output.")

# ── RIGHT PANEL ───────────────────────────────────────────────────────────────
with right:
    st.subheader("🔦 Evidence Highlights")

    if result and original_narrative:
        highlights = result.get("evidence_highlights", [])
        highlighted_html = highlight_narrative(original_narrative, highlights)

        st.markdown(
            f'<div style="font-family:monospace;font-size:13px;line-height:1.7;'
            f'background:#fafafa;border:1px solid #e0e0e0;border-radius:6px;'
            f'padding:16px;max-height:500px;overflow-y:auto">'
            f'{highlighted_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

        if highlights:
            st.markdown("**Signal legend:**")
            for h in highlights:
                signal_label = h["signal"].replace("_", " ").title()
                color = h.get("color", "#555")
                st.markdown(
                    f'<div style="font-size:12px;margin:4px 0">'
                    f'<span style="display:inline-block;width:12px;height:12px;'
                    f'background:{color};border-radius:2px;margin-right:6px"></span>'
                    f'<strong>{signal_label}</strong>: {h["explanation"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No domain signals highlighted in this narrative.")
    else:
        st.info("Evidence highlighting will appear here after triage.")
