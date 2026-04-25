import gradio as gr
import requests

BASE_MODEL_API = "https://your-base-model-api.com/generate"
TRAINED_MODEL_API = "https://your-trained-model-api.com/generate"

def call_base_model(prompt):
    try:
        response = requests.post(BASE_MODEL_API, json={"prompt": prompt})
        return response.json().get("output", "No response")
    except:
        return "Base model error"

def call_trained_model(prompt):
    try:
        response = requests.post(TRAINED_MODEL_API, json={"prompt": prompt})
        return response.json().get("output", "No response")
    except:
        return "Trained model error"

def compute_reward(output):
    score = 0
    if len(output) > 20:
        score += 1
    if "error" not in output.lower():
        score += 1
    return score


def full_pipeline(prompt):
    before_output = call_base_model(prompt)
    after_output = call_trained_model(prompt)

    before_reward = compute_reward(before_output)
    after_reward = compute_reward(after_output)

    improvement = after_reward - before_reward

    return (
        before_output,
        after_output,
        before_reward,
        after_reward,
        improvement
    )


ICON_COMPARE = """<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12h7v9H3zM14 3h7v18h-7z"></path></svg>"""
ICON_PROMPT = """<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 4h16v12H7l-3 4V4zm3 4h10v2H7V8zm0 4h7v2H7v-2z"></path></svg>"""
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
    background: #ffffff; /* Swapped to white */
    box-shadow: 0 4px 12px rgba(15, 18, 24, 0.04); /* Soft shadow */
}

.hero h1 {
    margin: 0;
    font-family: 'Manrope', sans-serif;
    font-size: clamp(1.7rem, 3vw, 2.4rem);
    font-weight: 800;
    display: flex;
    gap: 10px;
    align-items: center;
    color: #1f242b; /* Swapped to dark charcoal text */
}

.hero p {
    margin: 8px 0 0;
    color: #5a6370; /* Swapped to slate grey */
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

.gr-column { gap: 0 !important; }

/* Custom Button Styling */
.gradio-container button.primary {
    background: linear-gradient(180deg, #39414d 0%, #2a3038 100%) !important;
    border: 1px solid #1f252d !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 700 !important;
}

/* Center and resize the Run button */
#run-btn {
    max-width: 260px !important; /* Adjust this number to make it wider/narrower */
    margin: 24px auto !important; /* The 'auto' pushes it to the dead center */
    display: block !important;
    width: 100% !important;
}
#run-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 18px rgba(27, 33, 41, 0.4) !important;
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
  <h1>{ICON_COMPARE} Legacy Forge </h1>
  <p>Compare base model and trained model outputs with a side-by-side reward summary.</p>
</div>
"""
    )

    gr.Markdown(f"<div class=\"section-title\">{ICON_PROMPT} Prompt</div>")



    with gr.Row(equal_height=True):

        with gr.Column(elem_classes=["result-card"]): 
            gr.Markdown("<div class=\"input-title\">Task Prompt</div>")
            prompt_input = gr.Textbox(
                label="Task Prompt",
                placeholder="Type your problem here...",
                lines=3,
                show_label=False,
                container=False
            )


    run_button = gr.Button("Run Comparison", variant="primary", elem_id="run-btn")

    gr.Markdown(f"<div class=\"section-title\">{ICON_METRICS} Results</div>")

    with gr.Row(equal_height=True):
        with gr.Column(elem_classes=["result-card"]):
            gr.Markdown(f"<h3>{ICON_BASE} Base Model Output</h3>")
            before_output = gr.Textbox(show_label=False, lines=8, container=False)
        with gr.Column(elem_classes=["result-card"]):
            gr.Markdown(f"<h3>{ICON_TRAINED} Trained Model Output</h3>")
            after_output = gr.Textbox(show_label=False, lines=8, container=False)

    with gr.Row(equal_height=True):
        with gr.Column(elem_classes=["metric-box"]):
            gr.Markdown("<div class=\"metric-title\">Base Model Reward</div>")
            before_reward = gr.Number(show_label=False, precision=2, container=False)
        with gr.Column(elem_classes=["metric-box"]):
            gr.Markdown("<div class=\"metric-title\">Trained Model Reward</div>")
            after_reward = gr.Number(show_label=False, precision=2, container=False)
        with gr.Column(elem_classes=["metric-box"]):
            gr.Markdown("<div class=\"metric-title\">Improvement</div>")
            improvement = gr.Number(show_label=False, precision=2, container=False)

    run_button.click(
        fn=full_pipeline,
        inputs=prompt_input,
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
### Overview
- Base model performance compared with RL-trained model
- Objective reward comparison
- Measurable improvement signal
"""
    )

if __name__ == "__main__":
    demo.launch(theme=my_custom_theme, css=CUSTOM_CSS)