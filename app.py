"""Refer to https://github.com/abacaj/mpt-30B-inference/blob/main/download_model.py."""
# pylint: disable=invalid-name, missing-function-docstring, missing-class-docstring, redefined-outer-name, broad-except
import os
import time
from dataclasses import asdict, dataclass

import gradio as gr
from ctransformers import AutoConfig, AutoModelForCausalLM

from mcli import predict
from huggingface_hub import hf_hub_download
from loguru import logger

URL = os.getenv("URL", "")
MOSAICML_API_KEY = os.getenv("MOSAICML_API_KEY", "")
if URL is None:
    raise ValueError("URL environment variable must be set")
if MOSAICML_API_KEY is None:
    raise ValueError("git environment variable must be set")


def predict0(prompt, bot):
    # logger.debug(f"{prompt=}, {bot=}, {timeout=}")
    logger.debug(f"{prompt=}, {bot=}")
    try:
        user_prompt = prompt
        generator = generate(llm, generation_config, system_prompt, user_prompt.strip())
        print(assistant_prefix, end=" ", flush=True)

        response = ""
        for word in generator:
            print(word, end="", flush=True)
            response += word
        print("")
        logger.debug(f"{response=}")
    except Exception as exc:
        logger.error(exc)
        response = f"{exc=}"
    # bot = {"inputs": [response]}
    bot = [(prompt, response)]

    return prompt, bot


def download_mpt_quant(destination_folder: str, repo_id: str, model_filename: str):
    local_path = os.path.abspath(destination_folder)
    return hf_hub_download(
        repo_id=repo_id,
        filename=model_filename,
        local_dir=local_path,
        local_dir_use_symlinks=True,
    )


@dataclass
class GenerationConfig:
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    max_new_tokens: int
    seed: int
    reset: bool
    stream: bool
    threads: int
    stop: list[str]


def format_prompt(system_prompt: str, user_prompt: str):
    """format prompt based on: https://huggingface.co/spaces/mosaicml/mpt-30b-chat/blob/main/app.py"""

    system_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    user_prompt = f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
    assistant_prompt = f"<|im_start|>assistant\n"

    return f"{system_prompt}{user_prompt}{assistant_prompt}"


def generate(
    llm: AutoModelForCausalLM,
    generation_config: GenerationConfig,
    system_prompt: str,
    user_prompt: str,
):
    """run model inference, will return a Generator if streaming is true"""

    return llm(
        format_prompt(
            system_prompt,
            user_prompt,
        ),
        **asdict(generation_config),
    )


class Chat:
    default_system_prompt = "A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers."
    system_format = "<|im_start|>system\n{}<|im_end|>\n"

    def __init__(
        self, system: str = None, user: str = None, assistant: str = None
    ) -> None:
        if system is not None:
            self.set_system_prompt(system)
        else:
            self.reset_system_prompt()
        self.user = user if user else "<|im_start|>user\n{}<|im_end|>\n"
        self.assistant = (
            assistant if assistant else "<|im_start|>assistant\n{}<|im_end|>\n"
        )
        self.response_prefix = self.assistant.split("{}", maxsplit=1)[0]

    def set_system_prompt(self, system_prompt):
        # self.system = self.system_format.format(system_prompt)
        return system_prompt

    def reset_system_prompt(self):
        return self.set_system_prompt(self.default_system_prompt)

    def history_as_formatted_str(self, system, history) -> str:
        system = self.system_format.format(system)
        text = system + "".join(
            [
                "\n".join(
                    [
                        self.user.format(item[0]),
                        self.assistant.format(item[1]),
                    ]
                )
                for item in history[:-1]
            ]
        )
        text += self.user.format(history[-1][0])
        text += self.response_prefix
        # stopgap solution to too long sequences
        if len(text) > 4500:
            # delete from the middle between <|im_start|> and <|im_end|>
            # find the middle ones, then expand out
            start = text.find("<|im_start|>", 139)
            end = text.find("<|im_end|>", 139)
            while end < len(text) and len(text) > 4500:
                end = text.find("<|im_end|>", end + 1)
                text = text[:start] + text[end + 1 :]
        if len(text) > 4500:
            # the nice way didn't work, just truncate
            # deleting the beginning
            text = text[-4500:]

        return text

    def clear_history(self, history):
        return []

    def turn(self, user_input: str):
        self.user_turn(user_input)
        return self.bot_turn()

    def user_turn(self, user_input: str, history):
        history.append([user_input, ""])
        return user_input, history

    def bot_turn(self, system, history):
        conversation = self.history_as_formatted_str(system, history)
        assistant_response = call_inf_server(conversation)
        history[-1][-1] = assistant_response
        print(system)
        print(history)
        return "", history


def call_inf_server(prompt):
    try:
        response = predict(
            URL,
            {"inputs": [prompt], "temperature": 0.2, "top_p": 0.9, "output_len": 512},
            timeout=70,
        )
        # print(f'prompt: {prompt}')
        # print(f'len(prompt): {len(prompt)}')
        response = response["outputs"][0]
        # print(f'len(response): {len(response)}')
        # remove spl tokens from prompt
        spl_tokens = ["<|im_start|>", "<|im_end|>"]
        clean_prompt = prompt.replace(spl_tokens[0], "").replace(spl_tokens[1], "")

        # return response[len(clean_prompt) :]  # remove the prompt
        try:
            user_prompt = prompt
            generator = generate(llm, generation_config, system_prompt, user_prompt.strip())
            print(assistant_prefix, end=" ", flush=True)
            for word in generator:
                print(word, end="", flush=True)
            print("")
            response = word
        except Exception as exc:
            logger.error(exc)
            response = f"{exc=}"
        return response

    except Exception as e:
        # assume it is our error
        # just wait and try one more time
        print(e)
        time.sleep(1)
        response = predict(
            URL,
            {"inputs": [prompt], "temperature": 0.2, "top_p": 0.9, "output_len": 512},
            timeout=70,
        )
        # print(response)
        response = response["outputs"][0]
        return response[len(prompt) :]  # remove the prompt


logger.info("start dl")
_ = """full url: https://huggingface.co/TheBloke/mpt-30B-chat-GGML/blob/main/mpt-30b-chat.ggmlv0.q4_1.bin"""

repo_id = "TheBloke/mpt-30B-chat-GGML"
model_filename = "mpt-30b-chat.ggmlv0.q4_1.bin"
destination_folder = "models"

download_mpt_quant(destination_folder, repo_id, model_filename)

logger.info("done dl")

config = AutoConfig.from_pretrained("mosaicml/mpt-30b-chat", context_length=8192)
llm = AutoModelForCausalLM.from_pretrained(
    os.path.abspath("models/mpt-30b-chat.ggmlv0.q4_1.bin"),
    model_type="mpt",
    config=config,
)

system_prompt = "A conversation between a user and an LLM-based AI assistant named Local Assistant. Local Assistant gives helpful and honest answers."

generation_config = GenerationConfig(
    temperature=0.2,
    top_k=0,
    top_p=0.9,
    repetition_penalty=1.0,
    max_new_tokens=512,  # adjust as needed
    seed=42,
    reset=False,  # reset history (cache)
    stream=True,  # streaming per word/token
    threads=int(os.cpu_count() / 2),  # adjust for your CPU
    stop=["<|im_end|>", "|<"],
)

user_prefix = "[user]: "
assistant_prefix = "[assistant]: "

css = """
    .disclaimer {font-variant-caps: all-small-caps; font-size: xx-small;}
    .intro {font-size: x-small;}
"""

with gr.Blocks(
    theme=gr.themes.Soft(),
    css=css,
) as demo:
    with gr.Accordion("🎈 Info", open=False):
        gr.Markdown(
            """<h4><center>mosaicml mpt-30b-chat</center></h4>

            This demo is of [TheBloke/mpt-30B-chat-GGML](TheBloke/mpt-30B-chat-GGML.)

            It takes about >40 seconds to get a response.
            """,
            elem_classes="intro"
        )
    conversation = Chat()
    chatbot = gr.Chatbot().style(height=500)  # 500
    with gr.Row():
        with gr.Column():
            msg = gr.Textbox(
                label="Chat Message Box",
                placeholder="Chat Message Box",
                show_label=False,
            ).style(container=False)
        with gr.Column():
            with gr.Row():
                submit = gr.Button("Submit")
                stop = gr.Button("Stop", visible=False)
                clear = gr.Button("Clear", visible=False)
    with gr.Row(visible=False):
        with gr.Accordion("Advanced Options:", open=False):
            with gr.Row():
                with gr.Column(scale=2):
                    system = gr.Textbox(
                        label="System Prompt",
                        value=Chat.default_system_prompt,
                        show_label=False,
                    ).style(container=False)
                with gr.Column():
                    with gr.Row():
                        change = gr.Button("Change System Prompt")
                        reset = gr.Button("Reset System Prompt")
    # with gr.Row():
    with gr.Accordion("Disclaimer", open=False):
        gr.Markdown(
            "Disclaimer: MPT-30B can produce factually incorrect output, and should not be relied on to produce "
            "factually accurate information. MPT-30B was trained on various public datasets; while great efforts "
            "have been taken to clean the pretraining data, it is possible that this model could generate lewd, "
            "biased, or otherwise offensive outputs.",
            elem_classes=["disclaimer"],
        )
    with gr.Row(visible=False):
        gr.Markdown(
            "[Privacy policy](https://gist.github.com/samhavens/c29c68cdcd420a9aa0202d0839876dac)",
            elem_classes=["disclaimer"],
        )

    _ = """
    submit_event = msg.submit(
        fn=conversation.user_turn,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        fn=conversation.bot_turn,
        inputs=[system, chatbot],
        outputs=[msg, chatbot],
        queue=True,
    )
    submit_click_event = submit.click(
        fn=conversation.user_turn,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        # fn=conversation.bot_turn,
        inputs=[system, chatbot],
        outputs=[msg, chatbot],
        queue=True,
    )

    stop.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[submit_event, submit_click_event],
        queue=False,
    )
    clear.click(lambda: None, None, chatbot, queue=False).then(
        fn=conversation.clear_history,
        inputs=[chatbot],
        outputs=[chatbot],
        queue=False,
    )
    change.click(
        fn=conversation.set_system_prompt,
        inputs=[system],
        outputs=[system],
        queue=False,
    )
    reset.click(
        fn=conversation.reset_system_prompt,
        inputs=[],
        outputs=[system],
        queue=False,
    )
    # """

    msg.submit(
        # fn=conversation.user_turn,
        fn=predict0,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=True,
        show_progress="full",
        api_name="predict"
    )
    submit.click(
        # fn=conversation.user_turn,
        fn=predict0,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=True,
        show_progress="full",
    )

demo.queue(max_size=36, concurrency_count=14).launch(debug=True)
