"""
Marketing Budget Allocator — Streamlit App

Visualizes how an advertising budget should be optimally split across
marketing channels modeled with Hill response curves.

The budget slider lives inside the Plotly figure itself, so the visualization
updates fluidly as you drag (no Streamlit rerun on each tick).

Run with:
    streamlit run app.py
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Marketing Budget Allocator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        h1 { font-weight: 700; letter-spacing: -0.02em; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────
def hill(x, k, beta=1.0):
    x = np.asarray(x, dtype=float)
    return (x ** beta) / (k ** beta + x ** beta)

def optimal_allocation(B, ks):
    """β=1 optimal allocation. Drops highest-k channel when its share goes negative."""
    n = len(ks)
    out = [0.0] * n
    if B <= 0 or n == 0:
        return out
    indices = sorted(range(n), key=lambda i: ks[i])
    while indices:
        active_ks = [ks[i] for i in indices]
        sum_sqrt = sum(np.sqrt(k) for k in active_ks)
        sum_k = sum(active_ks)
        total = B + sum_k
        xs = [np.sqrt(active_ks[j]) / sum_sqrt * total - active_ks[j]
              for j in range(len(indices))]
        if min(xs) < 0:
            indices.pop()
            continue
        for j, i in enumerate(indices):
            out[i] = xs[j]
        return out
    out[min(range(n), key=lambda i: ks[i])] = B
    return out

def allocate(B, channels, active_flags):
    active_idx = [i for i, on in enumerate(active_flags) if on]
    out = [0.0] * len(channels)
    if not active_idx or B <= 0:
        return out
    if len(active_idx) == 1:
        out[active_idx[0]] = B
        return out
    sub_ks = [channels[i]["k"] for i in active_idx]
    sub_alloc = optimal_allocation(B, sub_ks)
    for j, i in enumerate(active_idx):
        out[i] = sub_alloc[j]
    return out

# ──────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────
DEFAULT_CHANNELS = [
    {"label": "Google Search", "k": 1500,  "beta": 1.0, "color": "#1f77b4"},
    {"label": "Meta",          "k": 8000,  "beta": 1.0, "color": "#d62728"},
    {"label": "Snapchat",      "k": 14000, "beta": 1.0, "color": "#2ca02c"},
]

X_MAX = 50_000
Y_SCALE = 12 * X_MAX
PROFIT_MAX = 1_500_000
N_BUDGET_STEPS = 101  # slider granularity (every $500 across 0..$50k)
CURVE_POINTS = 250    # x-grid resolution for response curves

# ──────────────────────────────────────────────────────────────────────────
# Sidebar — controls
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Channels")
    st.caption("Toggle channels and adjust their saturation parameters.")

    channels = []
    for i, default in enumerate(DEFAULT_CHANNELS):
        with st.expander(f"**{default['label']}**", expanded=True):
            active = st.checkbox("Active", value=True, key=f"active_{i}")
            k = st.number_input(
                "k (half-saturation, USD)",
                min_value=100, max_value=100_000,
                value=default["k"], step=100,
                key=f"k_{i}",
                help="The spend at which this channel reaches half its maximum revenue.",
            )
            channels.append({
                "label": default["label"], "k": k, "beta": default["beta"],
                "color": default["color"], "active": active,
            })

# ──────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────
st.title("Marketing Budget Allocator")
st.caption(
    "Drag the slider on the chart to see how the optimal budget split shifts "
    "across channels. Updates fluidly as you drag."
)

# ──────────────────────────────────────────────────────────────────────────
# Precompute all allocations across the budget range
# (cheap, vectorizable, ~100 evaluations of a 3-var optimization)
# ──────────────────────────────────────────────────────────────────────────
budget_steps = np.linspace(0, X_MAX, N_BUDGET_STEPS)
n_ch = len(channels)
active_flags = [c["active"] for c in channels]

allocations_per_B = np.array([
    allocate(B, channels, active_flags) for B in budget_steps
])  # shape: (N_BUDGET_STEPS, n_ch)
revenues_per_B = np.zeros_like(allocations_per_B)
for i, c in enumerate(channels):
    if c["active"]:
        revenues_per_B[:, i] = Y_SCALE * hill(allocations_per_B[:, i], c["k"], c["beta"])
total_revenue_per_B = revenues_per_B.sum(axis=1)
total_spend_per_B = allocations_per_B.sum(axis=1)
total_profit_per_B = total_revenue_per_B - total_spend_per_B

# Precompute response curves (full faded line per channel — same across all frames)
x_grid = np.linspace(0, X_MAX, CURVE_POINTS)
curve_y = {i: Y_SCALE * hill(x_grid, c["k"], c["beta"]) for i, c in enumerate(channels)}

# ──────────────────────────────────────────────────────────────────────────
# Build the unified figure: response curves + allocation bars + profit bar
# ──────────────────────────────────────────────────────────────────────────
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=3, cols=1,
    row_heights=[0.58, 0.25, 0.14],
    vertical_spacing=0.16,
    subplot_titles=("Response curves", "Allocation per channel", "Total profit"),
)

# Static traces (don't change with budget): faded full curves for each active channel
for i, c in enumerate(channels):
    if not c["active"]:
        continue
    fig.add_trace(
        go.Scatter(
            x=x_grid, y=curve_y[i],
            mode="lines",
            line=dict(color=c["color"], width=1.5, dash="dot"),
            opacity=0.35, showlegend=False, hoverinfo="skip",
        ),
        row=1, col=1,
    )

# Index where the dynamic (per-frame) traces start
n_static = sum(1 for c in channels if c["active"])

# Default frame index — pick budget = 10000 (closest step)
default_idx = int(np.argmin(np.abs(budget_steps - 10_000)))

def make_dynamic_traces(B_idx):
    """Build the per-frame traces for the given budget step."""
    alloc = allocations_per_B[B_idx]
    revs = revenues_per_B[B_idx]
    traces = []

    # Per-channel: highlighted curve segment + allocation marker
    for i, c in enumerate(channels):
        if not c["active"]:
            continue
        xa = alloc[i]
        mask = x_grid <= xa
        # Highlighted curve segment
        traces.append(go.Scatter(
            x=x_grid[mask], y=curve_y[i][mask],
            mode="lines",
            line=dict(color=c["color"], width=4),
            name=f"{c['label']} (k={c['k']:,})",
            showlegend=True,
            xaxis="x", yaxis="y",
            hovertemplate=(
                f"<b>{c['label']}</b><br>"
                "Spend: $%{x:,.0f}<br>Revenue: $%{y:,.0f}<extra></extra>"
            ),
        ))
        # Allocation marker
        traces.append(go.Scatter(
            x=[xa] if xa > 0 else [None],
            y=[Y_SCALE * hill(xa, c["k"], c["beta"])] if xa > 0 else [None],
            mode="markers",
            marker=dict(color=c["color"], size=12, line=dict(color="white", width=2)),
            showlegend=False,
            xaxis="x", yaxis="y",
            hovertemplate=(
                f"<b>{c['label']} — allocation</b><br>"
                "Spend: $%{x:,.0f}<br>Revenue: $%{y:,.0f}<extra></extra>"
            ),
        ))

    # Allocation bar — only show active channels
    active_channels_data = [(i, c) for i, c in enumerate(channels) if c["active"]]
    bar_x = [alloc[i] for i, _ in active_channels_data]
    bar_y = [c["label"] for _, c in active_channels_data]
    bar_colors = [c["color"] for _, c in active_channels_data]
    bar_text = [
        f"${alloc[i]:,.0f}" if alloc[i] > 0 else ""
        for i, _ in active_channels_data
    ]
    traces.append(go.Bar(
        x=bar_x, y=bar_y, orientation="h",
        marker=dict(color=bar_colors, opacity=0.85),
        text=bar_text, textposition="inside",
        textfont=dict(color="white", size=13),
        showlegend=False,
        xaxis="x2", yaxis="y2",
        hovertemplate="<b>%{y}</b><br>Spend: $%{x:,.0f}<extra></extra>",
    ))

    # Profit bar
    total_profit = total_profit_per_B[B_idx]
    display_profit = max(min(total_profit, PROFIT_MAX), -PROFIT_MAX * 0.05)
    profit_color = "#16a34a" if total_profit >= 0 else "#dc2626"
    traces.append(go.Bar(
        x=[display_profit], y=["Profit"], orientation="h",
        marker=dict(color=profit_color, opacity=0.9),
        text=[f"${total_profit:,.0f}"],
        textposition="inside" if abs(display_profit) > PROFIT_MAX * 0.2 else "outside",
        textfont=dict(
            color="white" if abs(display_profit) > PROFIT_MAX * 0.2 else profit_color,
            size=14,
        ),
        showlegend=False,
        xaxis="x3", yaxis="y3",
        hovertemplate="<b>Total profit</b>: $%{x:,.0f}<extra></extra>",
    ))

    return traces

# Add dynamic traces for the default frame
for tr in make_dynamic_traces(default_idx):
    if isinstance(tr, go.Bar):
        # Determine which subplot from xaxis attribute
        if tr.xaxis == "x2":
            fig.add_trace(tr, row=2, col=1)
        else:
            fig.add_trace(tr, row=3, col=1)
    else:
        fig.add_trace(tr, row=1, col=1)

# Build frames — one per budget step
frames = []
for B_idx in range(N_BUDGET_STEPS):
    dyn = make_dynamic_traces(B_idx)
    # Frame data must include all traces (static + dynamic) in the SAME order as the figure.
    # We reuse static traces unchanged.
    frame_data = list(fig.data[:n_static]) + dyn
    frames.append(go.Frame(
        name=str(B_idx),
        data=frame_data,
    ))
fig.frames = frames

# Slider configuration — tick labels at meaningful budget values only
tick_indices = [int(round(N_BUDGET_STEPS * frac))
                for frac in (0, 0.2, 0.4, 0.6, 0.8, 1.0)]
tick_indices = [min(i, N_BUDGET_STEPS - 1) for i in tick_indices]

slider_steps = []
for B_idx in range(N_BUDGET_STEPS):
    label = f"${int(budget_steps[B_idx]):,}" if B_idx in tick_indices else ""
    slider_steps.append(dict(
        method="animate",
        args=[[str(B_idx)], dict(
            mode="immediate",
            frame=dict(duration=0, redraw=True),
            transition=dict(duration=0),
        )],
        label=label,
    ))

# Layout
fig.update_layout(
    height=880,
    margin=dict(l=20, r=20, t=60, b=120),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    hovermode="closest",
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right", x=1,
        bgcolor="rgba(0,0,0,0)",
        itemclick=False,
        itemdoubleclick=False,
    ),
    sliders=[dict(
        active=default_idx,
        currentvalue=dict(
            prefix="",
            font=dict(size=18, color="#111", weight="bold"),
            xanchor="left",
        ),
        pad=dict(t=40, b=10),
        len=0.9, x=0.05, y=-0.02,
        steps=slider_steps,
        ticklen=4,
    )],
    annotations=list(fig.layout.annotations) + [
        dict(
            text="Budget size",
            xref="paper", yref="paper",
            x=0.5, y=-0.05,
            xanchor="center", yanchor="top",
            showarrow=False,
            font=dict(size=16),
        ),
    ],
    bargap=0.35,
)

# Axes
fig.update_xaxes(
    title_text="Spend (USD)", range=[0, X_MAX],
    tickformat="$,.0f", gridcolor="rgba(128,128,128,0.15)",
    zerolinecolor="rgba(128,128,128,0.3)", row=1, col=1,
)
fig.update_yaxes(
    title_text="Revenue (USD)",
    range=[-Y_SCALE * 0.02, Y_SCALE * 1.05],
    tickformat="$,.0f", gridcolor="rgba(128,128,128,0.15)",
    zerolinecolor="rgba(128,128,128,0.3)", row=1, col=1,
)
fig.update_xaxes(
    range=[0, X_MAX], tickformat="$,.0f",
    gridcolor="rgba(128,128,128,0.15)", row=2, col=1,
)
fig.update_yaxes(autorange="reversed", showgrid=False, row=2, col=1)
fig.update_xaxes(
    range=[-PROFIT_MAX * 0.03, PROFIT_MAX], tickformat="$,.0f",
    gridcolor="rgba(128,128,128,0.15)",
    zerolinecolor="rgba(128,128,128,0.5)", row=3, col=1,
)
fig.update_yaxes(showgrid=False, showticklabels=False, row=3, col=1)

st.plotly_chart(fig, use_container_width=True)