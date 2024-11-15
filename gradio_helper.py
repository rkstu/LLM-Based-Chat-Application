from typing import Callable, Literal
import gradio as gr
from uuid import uuid4


chinese_examples = [
    ["你好!"],
    ["你是谁?"],
    ["请介绍一下上海"],
    ["请介绍一下英特尔公司"],
    ["晚上睡不着怎么办？"],
    ["给我讲一个年轻人奋斗创业最终取得成功的故事。"],
    ["给这个故事起一个标题。"],
]

english_examples = [
    "Who won the latest Nobel Prizes, and for what achievements?",
    "What are the latest electric vehicle innovations?",
    "What new exoplanets have been discovered recently?",
    "What’s the latest news from the recent climate summit?",
    "Which AI advancements were announced this year?",
    "What are the recent breakthroughs in cancer research?",
    "How is the cryptocurrency market performing this month?",
    "What new policies are impacting global tech companies?",
    "What are the latest cybersecurity threats to businesses?",
    "What recent discoveries have been made in deep-sea exploration?"
]

japanese_examples = [
    ["こんにちは！調子はどうですか?"],
    ["OpenVINOとは何ですか?"],
    ["あなたは誰ですか?"],
    ["Pythonプログラミング言語とは何か簡単に説明してもらえますか?"],
    ["シンデレラのあらすじを一文で説明してください。"],
    ["コードを書くときに避けるべきよくある間違いは何ですか?"],
    ["人工知能と「OpenVINOの利点」について100語程度のブログ記事を書いてください。"],
]


def get_uuid():
    """
    Universal unique identifier for thread
    """
    return str(uuid4())

def handle_user_message(message, history):
    """
    Callback function for updating user messages in interface on submit button click
    """
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]

def make_demo(
    run_fn: Callable,
    title: str = "OpenVINO Chatbot",
    language: Literal["English", "Chinese", "Japanese"] = "English"
):
    # Define examples based on the selected language
    examples = (
        chinese_examples if language == "Chinese" 
        else japanese_examples if language == "Japanese" 
        else english_examples
    )

    with gr.Blocks(
        theme=gr.themes.Soft(),
        css=".disclaimer {font-variant-caps: all-small-caps;}"
    ) as demo:
        conversation_id = gr.State(get_uuid)  # Ensure get_uuid is defined elsewhere
        gr.Markdown(f"<h1><center>{title}</center></h1>")
        chatbot = gr.Chatbot(height=500)

        # User message input
        with gr.Row():
            with gr.Column():
                msg = gr.Textbox(
                    label="Chat Message Box",
                    placeholder="Chat Message Box",
                    show_label=False,
                    container=False,
                )
            with gr.Column():
                submit = gr.Button("Submit")
                clear = gr.Button("Clear")

        # Advanced options for the chat
        with gr.Row():
            with gr.Accordion("Advanced Options:", open=False):
                temperature = gr.Slider(
                    label="Temperature",
                    value=0.1,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    interactive=True,
                    info="Higher values produce more diverse outputs",
                )
                top_p = gr.Slider(
                    label="Top-p (nucleus sampling)",
                    value=1.0,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    interactive=True,
                    info=("Sample from the smallest possible set of tokens whose cumulative probability exceeds top_p. "
                          "Set to 1 to disable and sample from all tokens."),
                )
                top_k = gr.Slider(
                    label="Top-k",
                    value=50,
                    minimum=0,
                    maximum=200,
                    step=1,
                    interactive=True,
                    info="Sample from a shortlist of top-k tokens — 0 to disable and sample from all tokens.",
                )
                repetition_penalty = gr.Slider(
                    label="Repetition Penalty",
                    value=1.1,
                    minimum=1.0,
                    maximum=2.0,
                    step=0.1,
                    interactive=True,
                    info="Penalize repetition — 1.0 to disable.",
                )

        # Example messages
        gr.Examples(examples, inputs=msg, label="Click on any example and press the 'Submit' button")

        # Submit message event
        submit_event = msg.submit(
            fn=handle_user_message,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False,
        ).then(
            fn=run_fn,
            inputs=[chatbot, temperature, top_p, top_k, repetition_penalty, conversation_id],
            outputs=chatbot,
            queue=True,
        )

        # Submit button click event
        submit.click(
            fn=handle_user_message,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False,
        ).then(
            fn=run_fn,
            inputs=[chatbot, temperature, top_p, top_k, repetition_penalty, conversation_id],
            outputs=chatbot,
            queue=True,
        )

        # Clear chat button functionality
        clear.click(lambda: None, None, chatbot, queue=False)

    return demo