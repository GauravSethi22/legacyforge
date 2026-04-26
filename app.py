import gradio as gr
import plotly.graph_objects as go
import numpy as np


TRAINING_REWARDS_BASE = np.array([
    
    0.0, 0.7, -0.1, -0.8, -0.1, -0.8, -0.1, -0.1, -0.1, -0.1, -0.1, -0.8, -0.1, -0.8, -0.1, -0.1, -0.1, -0.1, -0.1, -2.8,
    
    0.0, -0.6, -0.1, -0.1, -0.1, -0.1, -0.6, -0.1, -0.1, -0.1, -0.1, -0.6, -0.1, -0.1, -0.1, -0.1, -0.6, -0.1, -0.1, -2.1,
    
    0.0, 0.7, -0.1, -0.8, -0.1, -0.1, -0.8, -0.1, -0.8, -0.1, -0.1, -0.1, -0.1, -0.1, -0.8, -0.1, -0.8, -0.1, -0.1, -2.1,
    
    0.0, -0.6, -0.1, 0.6, -0.1, -0.4, -0.1, -0.4, -0.1, -0.4, -0.1, -0.4, -0.1, -0.4, -0.1, -0.4, -0.1, -0.4, -0.1, -0.4,
    
    0.0, -0.6, -0.1, -0.1, -0.1, -0.1, -0.6, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -2.1
])

TRAINING_REWARDS_TRAINED = np.array([
    -0.1, -0.1, -1.375, -0.1, -1.55, -0.1, -0.175, 1.0, 1.0, -0.1, 
    -0.1, -0.1, -0.1, -0.1, -0.275, -1.725, -0.1, -0.275, -0.1, -0.825, 
    -0.5, -0.1, -0.775, 0.0, 0.45, -0.1, 0.45, -0.1, -0.1, -0.1, 
    -0.1, -0.1, -0.1, -0.1, -0.1, -0.775, -0.1, -1.275, -0.1, -0.1, 
    0.5, -0.1, 0.45, -0.1, -0.1, -0.1, -2.05, -0.1, -0.1, -0.1, 
    -0.1, 0.325, -1.275, 1.0, -0.1, -0.1, -0.1, -0.275, -0.1, -0.1, 
    -0.775, -0.1, -0.1, 0.6, -0.1, -0.775, -0.175, -0.1, -0.1, -0.775, 
    1.0, 0.5, -0.1, -0.1, 0.5, -0.1, -0.1, -0.1, -0.1, -0.1, 
    1.0, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 0.5, 0.5, 
    0.5, -0.1, -1.775, -0.1, -0.1, -0.1, -0.775, -1.275, -0.1, 0.5, 
    -0.1, -0.375, -0.1, -0.1, 1.0, 0.5, 0.5, -0.775, -0.1, -0.1, 
    -0.05, -0.1, 1.0, -0.775, 0.5, -0.1, -2.55, -0.1, -0.95, -0.1, 
    -0.175, -0.1, -0.1, -1.325, -0.1, 0.5, 0.5, 2.1, 1.0, -0.1, 
    0.5, -0.1, 1.1, -0.05, -0.1, -0.1, 1.0, -0.05, -0.275, 0.5, 
    -0.1, -0.1, -0.1, -0.1, -0.1, 1.0, -0.1, -0.1, -0.1, -0.1
])

def create_plot(data_type="base"):
    """Converts the real data arrays into Plotly graphs."""
    if data_type == "base":
        y = TRAINING_REWARDS_BASE
        steps = np.arange(1, len(y) + 1)
        line_color = "#ef4444" 
        title = "Pre-RL Baseline Trajectory (Llama-3.1 Base)"
        x_title = "Action Steps Across Episodes (100 Steps)"
    else:
        y = TRAINING_REWARDS_TRAINED
        steps = np.arange(1, len(y) + 1)
        line_color = "#22c55e" 
        title = "Post-RL Optimized Trajectory (150 Steps)"
        x_title = "Training Steps (Agent Environment)"

    fig = go.Figure(data=go.Scatter(x=steps, y=y, mode='lines', line=dict(color=line_color, width=2.5)))
    
    fig.update_layout(
        title=title,
        title_font=dict(family="Manrope", size=14, color="#1f242b"),
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        xaxis=dict(title=x_title, showgrid=True, gridcolor="#ebeef5", zeroline=False),
        yaxis=dict(title="Step Reward", showgrid=True, gridcolor="#ebeef5", zeroline=False)
    )
    return fig

def full_pipeline(prompt=None):
    before_plot = create_plot("base")
    after_plot = create_plot("trained")
    before_reward = -5.62
    after_reward = -3.20
    improvement = after_reward - before_reward

    return (
        before_plot,
        after_plot,
        before_reward,
        after_reward,
        improvement
    )


ICON_COMPARE = """<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12h7v9H3zM14 3h7v18h-7z"></path></svg>"""
ICON_BASE = """<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 6h16v12H4zM7 9h10v2H7zm0 4h6v2H7z"></path></svg>"""
ICON_TRAINED = """<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2l2.6 5.3 5.8.8-4.2 4.1 1 5.8L12 15.8 6.8 18l1-5.8L3.6 8.1l5.8-.8z"></path></svg>"""
ICON_METRICS = """<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M5 20h14v2H3V2h2v18zm2-2V9h3v9H7zm5 0V5h3v13h-3zm5 0v-6h3v6h-3z"></path></svg>"""

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&family=Source+Sans+3:wght@400;500;600&display=swap');

body, .gradio-container {
    font-family: 'Source Sans 3', sans-serif !important;
}

.gradio-container {
    padding: 28px 20px 22px !important;
}

.hero {
    border: 1px solid #dfe5ed;
    border-radius: 18px;
    padding: 24px;
    background: #ffffff; 
    box-shadow: 0 4px 12px rgba(15, 18, 24, 0.04); 
}

.hero h1 {
    margin: 0;
    font-family: 'Manrope', sans-serif;
    font-size: clamp(1.7rem, 3vw, 2.4rem);
    font-weight: 800;
    display: flex;
    gap: 10px;
    align-items: center;
    color: #1f242b; 
}

.hero p {
    margin: 8px 0 0;
    color: #5a6370; 
    font-size: 1rem;
    font-weight: 500;
}

.section-title {
    margin: 22px 0 10px;
    font-family: 'Manrope', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 8px;
    color: #1f242b;
}

.input-title, .metric-title {
    font-family: 'Manrope', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.9px;
    text-transform: uppercase;
    color: #5a6370;
    margin: 0 0 10px;
}

.result-card {
    border: 1px solid #dfe5ed;
    border-radius: 14px;
    padding: 16px;
    background: #ffffff;
    box-shadow: 0 4px 12px rgba(15, 18, 24, 0.04);
}

.result-card h3 {
    margin: 0 0 12px 0;
    font-family: 'Manrope', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 8px;
    color: #1f242b;
}

.metric-box {
    border: 1px solid #ebeef5 !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

.icon {
    width: 22px;
    height: 22px;
    fill: currentColor;
    flex-shrink: 0;
    overflow: visible;
}
"""

my_custom_theme = gr.themes.Base().set(
    background_fill_primary="#f4f6f8",
    background_fill_primary_dark="#f4f6f8",
    input_background_fill="#ffffff",
    input_background_fill_dark="#ffffff",
    body_text_color="#1e2329",
    body_text_color_dark="#1e2329",
    input_border_color="#cbd5e1",
    input_border_color_dark="#cbd5e1",
    input_border_color_focus="#3b82f6",
    input_border_color_focus_dark="#3b82f6",
    block_background_fill="transparent",
    block_background_fill_dark="transparent",
    block_border_width="0px"
)


with gr.Blocks(fill_width=True) as demo:

    gr.Markdown(
        f"""
<div class="hero">
  <h1>{ICON_COMPARE} Legacy Forge</h1>
  <p>Tracking the performance improvements between the baseline Llama 3.1 model and the optimized, multi-stage agent.</p>
</div>
"""
    )

    gr.Markdown(f"<div class=\"section-title\">{ICON_METRICS} Results</div>")

    with gr.Row(equal_height=True):
        with gr.Column(elem_classes=["result-card"]):
            gr.Markdown(f"<h3>{ICON_BASE}  Baseline Reward Graph</h3>")
            before_output = gr.Plot(show_label=False, container=False) 
            
        with gr.Column(elem_classes=["result-card"]):
            gr.Markdown(f"<h3>{ICON_TRAINED} RL Optimized Reward Graph</h3>")
            after_output = gr.Plot(show_label=False, container=False)

    with gr.Row(equal_height=True):
        with gr.Column(elem_classes=["metric-box"]):
            gr.Markdown("<div class=\"metric-title\">Baseline Total Reward </div>")
            before_reward = gr.Number(show_label=False, precision=2, container=False)
            
        with gr.Column(elem_classes=["metric-box"]):
            gr.Markdown("<div class=\"metric-title\">RL Trained Reward</div>")
            after_reward = gr.Number(show_label=False, precision=2, container=False)
            
        with gr.Column(elem_classes=["metric-box"]):
            gr.Markdown("<div class=\"metric-title\">Total Improvement</div>")
            improvement = gr.Number(show_label=False, precision=2, container=False)

    demo.load(
        fn=full_pipeline,
        inputs=None,
        outputs=[
            before_output,
            after_output,
            before_reward,
            after_reward,
            improvement
        ]
    )

    gr.Markdown(
        """
---
### Execution Summary
This dashboard tracks the behavioral alignment of the model through Reinforcement Learning.

* **Pre-RL Baseline:** Visualizes the unaligned model's initial state-space exploration across 5 episodes (100 steps). The agent remains largely stagnant, failing to consistently secure high-value outcomes.
* **Post-RL Convergence:** Maps the telemetry of the trained model over 150 optimization steps. The graph highlights the transition from initial variance to an exploitative strategy, demonstrating the agent's ability to maximize the reward function.
* **Policy Advantage:** The delta between the exact initial baseline summary distribution and the final converged RL state (+2.42), confirming successful algorithmic alignment.
"""
    )

if __name__ == "__main__":
    demo.launch(theme=my_custom_theme, css=CUSTOM_CSS)