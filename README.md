# LLM-Based Chat Application

This is a language model-based chat application that provides users with relevant, up-to-date, and easy-to-understand responses by leveraging web search and a powerful language model. The basic flow involves performing a web search using DuckDuckGo, followed by generating a response using the Qwen2.5-0.5b-Instruct language model.

## Features

- **Web Search**: Performs real-time searches using DuckDuckGo to retrieve the latest relevant information.
- **Language Model Integration**: Uses the `qwen2.5-0.5b-instruct` model for generating accurate and coherent responses.
- **Simple Interface**: Built with Gradio to offer an easy-to-use web interface for users to interact with the model.
- **Efficient and Scalable**: Optimized for performance using Intel-specific optimizations via `optimum-intel` and other acceleration libraries.

## Model Used
- Language Model: qwen2.5-0.5b-instruct
- A powerful and efficient model that generates human-like responses based on the query.

## Application Flow
- **User Query**: The user inputs a question or request.
- **Web Search**: The system performs a web search using DuckDuckGo to gather recent and relevant information.
- **Response Generation**: The gathered data is used to generate a clear, relevant, and easy-to-understand response using the qwen2.5-0.5b-instruct language model.
- **User Response**: The application returns the generated response to the user.

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


## License
- This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Acknowledgements
- **OpenVINO**: For efficient acceleration of the language model.
- **DuckDuckGo**: For providing real-time web search capabilities.
- **Hugging Face**: For providing powerful transformer models and tools.
