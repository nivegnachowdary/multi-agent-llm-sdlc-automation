import gradio as gr
from utils.langgraph_pipeline import run_pipeline_and_save

def handle_run(prompt):
    chat_log, zip_path = run_pipeline_and_save(prompt)
    return chat_log, zip_path

with gr.Blocks() as demo:
    gr.Markdown("# 🔧 Multi-Agent UI Generator")
    input_box   = gr.Textbox(lines=4, label="Enter your product idea prompt")
    run_btn     = gr.Button("Generate Website")
    chatbox     = gr.Chatbot(label="Agent Conversation Log", type="messages")
    file_output = gr.File(label="Download UI ZIP")

    run_btn.click(
        fn=handle_run,
        inputs=[input_box],
        outputs=[chatbox, file_output],
    )

demo.launch()
