# LLM-Based Chat Application

This is a language model-based chat application that provides users with relevant, up-to-date, and easy-to-understand responses by leveraging web search and a powerful language model. The basic flow involves performing a web search using DuckDuckGo, followed by generating a response using the Qwen2.5-0.5b-Instruct language model.

## Features

- **Web Search**: Performs real-time searches using DuckDuckGo to retrieve the latest relevant information.
- **Language Model Integration**: Uses the `qwen2.5-0.5b-instruct` model for generating accurate and coherent responses.
- **Simple Interface**: Built with Gradio to offer an easy-to-use web interface for users to interact with the model.
- **Efficient and Scalable**: Optimized for performance using Intel-specific optimizations via `optimum-intel` and other acceleration libraries.

## Dependencies

The following dependencies are required for running this application:

- `openvino>=2024.2.0`
- `openvino-tokenizers[transformers]`
- `torch>=2.1`
- `datasets`
- `duckduckgo-search`
- `langchain-community`
- `accelerate`
- `gradio>=4.19`
- `onnx<=1.16.1` (For Windows platform `sys_platform=='win32'`)
- `einops`
- `transformers>=4.43.1`
- `transformers_stream_generator`
- `tiktoken`
- `bitsandbytes`
- `optimum-intel` (installed via `git+https://github.com/huggingface/optimum-intel.git`)
- `nncf` (installed via `git+https://github.com/openvinotoolkit/nncf.git`)

## Installation

To install the required dependencies, you can use the following command:

```bash
pip install -r requirements.txt
```


---
title: Llm Chatbot
emoji: 👀
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 5.4.0
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference