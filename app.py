# app.py
import os
from pathlib import Path
import torch
from threading import Event, Thread
from typing import List, Tuple

# Importing necessary packages
from transformers import AutoConfig, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from langchain_community.tools import DuckDuckGoSearchRun
from optimum.intel.openvino import OVModelForCausalLM
import openvino as ov
import openvino.properties as props
import openvino.properties.hint as hints
import openvino.properties.streams as streams

from gradio_helper import make_demo  # UI logic import
from llm_config import SUPPORTED_LLM_MODELS

# Model configuration setup
max_new_tokens = 256
model_language_value = "English"
model_id_value = 'qwen2.5-0.5b-instruct'
prepare_int4_model_value = True
enable_awq_value = False
device_value = 'CPU'
model_to_run_value = 'INT4'
pt_model_id = SUPPORTED_LLM_MODELS[model_language_value][model_id_value]["model_id"]
pt_model_name = model_id_value.split("-")[0]
int4_model_dir = Path(model_id_value) / "INT4_compressed_weights"
int4_weights = int4_model_dir / "openvino_model.bin"

model_configuration = SUPPORTED_LLM_MODELS[model_language_value][model_id_value]
model_name = model_configuration["model_id"]
start_message = model_configuration["start_message"]
history_template = model_configuration.get("history_template")
has_chat_template = model_configuration.get("has_chat_template", history_template is None)
current_message_template = model_configuration.get("current_message_template")
stop_tokens = model_configuration.get("stop_tokens")
tokenizer_kwargs = model_configuration.get("tokenizer_kwargs", {})

# Model loading
core = ov.Core()
ov_config = {
    hints.performance_mode(): hints.PerformanceMode.LATENCY,
    streams.num(): "1",
    props.cache_dir(): ""
}
tok = AutoTokenizer.from_pretrained(int4_model_dir, trust_remote_code=True)
ov_model = OVModelForCausalLM.from_pretrained(
    int4_model_dir,
    device=device_value,
    ov_config=ov_config,
    config=AutoConfig.from_pretrained(int4_model_dir, trust_remote_code=True),
    trust_remote_code=True,
)

# Define stopping criteria for specific token sequences
class StopOnTokens(StoppingCriteria):
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return any(input_ids[0][-1] == stop_id for stop_id in self.token_ids)

if stop_tokens is not None:
    if isinstance(stop_tokens[0], str):
        stop_tokens = tok.convert_tokens_to_ids(stop_tokens)
    stop_tokens = [StopOnTokens(stop_tokens)]

# Helper function for partial text update
def default_partial_text_processor(partial_text: str, new_text: str) -> str:
    return partial_text + new_text

text_processor = model_configuration.get("partial_text_processor", default_partial_text_processor)

# Convert conversation history to tokens based on model template
def convert_history_to_token(history: List[Tuple[str, str]]):
    if pt_model_name == "baichuan2":
        system_tokens = tok.encode(start_message)
        history_tokens = []
        for old_query, response in history[:-1]:
            round_tokens = [195] + tok.encode(old_query) + [196] + tok.encode(response)
            history_tokens = round_tokens + history_tokens
        input_tokens = system_tokens + history_tokens + [195] + tok.encode(history[-1][0]) + [196]
        input_token = torch.LongTensor([input_tokens])
    elif history_template is None or has_chat_template:
        messages = [{"role": "system", "content": start_message}]
        for idx, (user_msg, model_msg) in enumerate(history):
            if idx == len(history) - 1 and not model_msg:
                messages.append({"role": "user", "content": user_msg})
                break
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if model_msg:
                messages.append({"role": "assistant", "content": model_msg})
        input_token = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt")
    else:
        text = start_message + "".join(
            [history_template.format(num=round, user=item[0], assistant=item[1]) for round, item in enumerate(history[:-1])]
        )
        text += current_message_template.format(num=len(history) + 1, user=history[-1][0], assistant=history[-1][1])
        input_token = tok(text, return_tensors="pt", **tokenizer_kwargs).input_ids
    return input_token

# Initialize search tool
search = DuckDuckGoSearchRun()

# Determine if a search is needed based on the query
def should_use_search(query: str) -> bool:
    search_keywords = ["latest", "news", "update", "which", "who", "what", "when", "why", "how", "recent", "current",
                      "announcement", "bulletin", "report", "brief", "insight", "disclosure", "update", 
                        "release", "memo", "headline", "current", "ongoing", "fresh", "upcoming", "immediate", 
                        "recently", "new", "now", "in-progress", "inquiry", "query", "ask", "investigate", 
                        "explore", "seek", "clarify", "confirm", "discover", "learn", "describe", "define", 
                        "illustrate", "outline", "interpret", "expound", "detail", "summarize", "elucidate", 
                        "break down", "outcome", "effect", "consequence", "finding", "achievement", "conclusion", 
                        "product", "performance", "resolution"
                      ]
    return any(keyword in query.lower() for keyword in search_keywords)

# Construct the prompt with optional search context
def construct_model_prompt(user_query: str, search_context: str, history: List[Tuple[str, str]]) -> str:
    instructions = (
        "Based on the information provided below, deliver an accurate, concise, and easily understandable answer. If relevant information is missing, draw on your general knowledge and mention the absence of specific details."
    )
    prompt = f"{instructions}\n\n{search_context if search_context else ''}\n\n{user_query} ?\n\n"
    return prompt

# Fetch search results for a query
def fetch_search_results(query: str) -> str:
    search_results = search.invoke(query)
    print("Search results:", search_results)  # Optional: Debugging output
    return f"Relevant and recent information:\n{search_results}"

# Main chatbot function
def bot(history, temperature, top_p, top_k, repetition_penalty, conversation_id):
    user_query = history[-1][0]
    search_context = fetch_search_results(user_query) if should_use_search(user_query) else ""
    prompt = construct_model_prompt(user_query, search_context, history)
    input_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=2500).input_ids if search_context else convert_history_to_token(history)

    # Limit input length to avoid exceeding token limit
    if input_ids.shape[1] > 2000:
        history = [history[-1]]

    # Configure response streaming
    streamer = TextIteratorStreamer(tok, timeout=4600.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": temperature > 0.0,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "streamer": streamer,
        "stopping_criteria": StoppingCriteriaList(stop_tokens) if stop_tokens is not None else None,
    }

    # Signal completion
    stream_complete = Event()
    def generate_and_signal_complete():
        try:
            ov_model.generate(**generate_kwargs)
        except RuntimeError as e:
            # Check if the error message indicates the request was canceled
            if "Infer Request was canceled" in str(e):
                print("Generation request was canceled.")
            else:
                # If it's a different RuntimeError, re-raise it
                raise e
        finally:
            # Signal completion of the stream
            stream_complete.set()

    t1 = Thread(target=generate_and_signal_complete)
    t1.start()

    partial_text = ""
    for new_text in streamer:
        partial_text = text_processor(partial_text, new_text)
        history[-1] = (user_query, partial_text)
        yield history

def request_cancel():
    ov_model.request.cancel()

# Gradio setup and launch
demo = make_demo(run_fn=bot, title=f"OpenVINO Search & Reasoning Chatbot", language=model_language_value)
if __name__ == "__main__":
    demo.launch(debug=True, share=True, server_name="0.0.0.0", server_port=7860)
