import gradio as gr 
from main import llm_response


def predict(message, history):
    # output = message # debug mode
    output = str(llm_response(message)).replace("\n", "<br/>")
    return output

demo = gr.ChatInterface(
    predict,
    title = f' Open-Source LLM Question Answering'
)

demo.queue()
demo.launch(server_name="0.0.0.0", server_port=7868)