"""Refer to https://github.com/abacaj/mpt-30B-inference."""
# pylint: disable=invalid-name, missing-function-docstring, missing-class-docstring, redefined-outer-name, broad-except, line-too-long
import os
import time
from dataclasses import asdict, dataclass
from types import SimpleNamespace

from about_time import about_time
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

ns = SimpleNamespace(response="")


def predict0(prompt, bot):
    # logger.debug(f"{prompt=}, {bot=}, {timeout=}")
    _ = ("", "")
    bot.append(_)
    logger.debug(f"{prompt=}, {bot=}")
    ns.response = ""
    with about_time() as atime:
        try:
            # user_prompt = prompt
            generator = generate(llm, generation_config, system_prompt, prompt.strip())
            print(assistant_prefix, end=" ", flush=True)

            response = ""
            buff.update(value="diggin...")
            if bot:
                bot[-1] = [(prompt, "diggin...")]
            else:
                bot = [(prompt, "diggin...")]
            for word in generator:
                print(word, end="", flush=True)
                response += word
                ns.response = response
                buff.update(value=response)
                bot[-1] = [(prompt, response)]
            print("")
            logger.debug(f"{response=}")
        except Exception as exc:
            logger.error(exc)
            response = f"{exc=}"

    # bot = {"inputs": [response]}
    _ = (
        f"(time elapsed: {atime.duration_human}, "
        f"{atime.duration/(len(prompt) + len(response))} s/char)"
    )
    bot[-1] = [(prompt, f"{response} {_}")]

    return prompt, bot


def predict_api(prompt):
    logger.debug(f"{prompt=}")
    ns.response = ""
    try:
        # user_prompt = prompt
        generator = generate(llm, generation_config, system_prompt, prompt.strip())
        print(assistant_prefix, end=" ", flush=True)

        response = ""
        buff.update(value="diggin...")
        for word in generator:
            print(word, end="", flush=True)
            response += word
            ns.response = response
            buff.update(value=response)
        print("")
        logger.debug(f"{response=}")
    except Exception as exc:
        logger.error(exc)
        response = f"{exc=}"
    # bot = {"inputs": [response]}
    # bot = [(prompt, response)]

    return response


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
    assistant_prompt = "<|im_start|>assistant\n"

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
            generator = generate(
                llm, generation_config, system_prompt, user_prompt.strip()
            )
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

# https://huggingface.co/TheBloke/mpt-30B-chat-GGML
_ = """
mpt-30b-chat.ggmlv0.q4_0.bin 	q4_0 	4 	16.85 GB 	19.35 GB 	4-bit.
mpt-30b-chat.ggmlv0.q4_1.bin 	q4_1 	4 	18.73 GB 	21.23 GB 	4-bit. Higher accuracy than q4_0 but not as high as q5_0. However has quicker inference than q5 models.
mpt-30b-chat.ggmlv0.q5_0.bin 	q5_0 	5 	20.60 GB 	23.10 GB
mpt-30b-chat.ggmlv0.q5_1.bin 	q5_1 	5 	22.47 GB 	24.97 GB
mpt-30b-chat.ggmlv0.q8_0.bin 	q8_0 	8 	31.83 GB 	34.33 GB
"""
model_filename = "mpt-30b-chat.ggmlv0.q4_1.bin"
destination_folder = "models"

download_mpt_quant(destination_folder, repo_id, model_filename)

logger.info("done dl")

config = AutoConfig.from_pretrained("mosaicml/mpt-30b-chat", context_length=8192)
llm = AutoModelForCausalLM.from_pretrained(
    os.path.abspath(f"models/{model_filename}"),
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
    .importantButton {
        background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
        border: none !important;
    }
    .importantButton:hover {
        background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
        border: none !important;
    }
    .disclaimer {font-variant-caps: all-small-caps; font-size: xx-small;}
    .xsmall {font-size: x-small;}
"""

with gr.Blocks(
    title="mpt-30b-chat-ggml",
    theme=gr.themes.Soft(text_size="sm"),
    css=css,
) as block:
    with gr.Accordion("🎈 Info", open=False):
        gr.HTML(
            """<center><a href="https://huggingface.co/spaces/mikeee/mpt-30b-chat?duplicate=true"><img src="https://bit.ly/3gLdBN6" alt="Duplicate"></a> and spin a CPU UPGRADE to avoid the queue</center>"""
        )
        gr.Markdown(
            """<h4><center>mpt-30b-chat-ggml (q4_1)</center></h4>

            This demo is of [TheBloke/mpt-30B-chat-GGML](https://huggingface.co/TheBloke/mpt-30B-chat-GGML).

            Try to refresh the browser and try again when occasionally errors occur.

            It takes about >40 seconds to get a response. Restarting the space takes about 5 minutes if the space is asleep due to inactivity. If the space crashes for some reason, it will also take about 5 minutes to restart. You need to refresh the browser to reload the new space.
            """,
            elem_classes="xsmall",
        )
    conversation = Chat()
    chatbot = gr.Chatbot().style(height=700)  # 500
    buff = gr.Textbox(show_label=False)
    with gr.Row():
        with gr.Column():
            msg = gr.Textbox(
                label="Chat Message Box",
                placeholder="Ask me anything (press Enter or click Submit to send)",
                show_label=False,
            ).style(container=False)
        with gr.Column():
            with gr.Row():
                submit = gr.Button("Submit", elem_classes="xsmall")
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

    with gr.Accordion("Example inputs", open=True):
        etext = """In America, where cars are an important part of the national psyche, a decade ago people had suddenly started to drive less, which had not happened since the oil shocks of the 1970s. """
        examples = gr.Examples(
            examples=[
                ["Explain the plot of Cinderella in a sentence."],
                [
                    "How long does it take to become proficient in French, and what are the best methods for retaining information?"
                ],
                ["What are some common mistakes to avoid when writing code?"],
                ["Build a prompt to generate a beautiful portrait of a horse"],
                ["Suggest four metaphors to describe the benefits of AI"],
                ["Write a pop song about leaving home for the sandy beaches."],
                ["Write a summary demonstrating my ability to tame lions"],
                ["鲁迅和周树人什么关系 说中文"],
                ["鲁迅和周树人什么关系"],
                ["鲁迅和周树人什么关系 用英文回答"],
                ["从前有一头牛，这头牛后面有什么？"],
                ["正无穷大加一大于正无穷大吗？"],
                ["正无穷大加正无穷大大于正无穷大吗？"],
                ["-2的平方根等于什么"],
                ["树上有5只鸟，猎人开枪打死了一只。树上还有几只鸟？"],
                ["树上有11只鸟，猎人开枪打死了一只。树上还有几只鸟？提示：需考虑鸟可能受惊吓飞走。"],
                ["以红楼梦的行文风格写一张委婉的请假条。不少于320字。"],
                [f"{etext} 翻成中文，列出3个版本"],
                [f"{etext} \n 翻成中文，保留原意，但使用文学性的语言。不要写解释。列出3个版本"],
                ["js 判断一个数是不是质数"],
                ["js 实现python 的 range(10)"],
                ["js 实现python 的 [*(range(10)]"],
                ["假定 1 + 2 = 4, 试求 7 + 8"],
                ["Erkläre die Handlung von Cinderella in einem Satz."],
                ["Erkläre die Handlung von Cinderella in einem Satz. Auf Deutsch"],
            ],
            inputs=[msg],
            examples_per_page=40,
        )

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
        api_name="predict",
    )
    submit.click(
        # fn=conversation.user_turn,
        fn=predict0,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=True,
        show_progress="full",
    )

    # update buff Textbox, every: units in seconds)
    # https://huggingface.co/spaces/julien-c/nvidia-smi/discussions
    # does not work
    # AttributeError: 'Blocks' object has no attribute 'run_forever'
    # block.run_forever(lambda: ns.response, None, [buff], every=1)

    with gr.Accordion("For Chat/Translation API", open=False, visible=False):
        input_text = gr.Text()
        api_btn = gr.Button("Go", variant="primary")
        out_text = gr.Text()
    api_btn.click(
        predict_api,
        input_text,
        out_text,
        # show_progress="full",
        api_name="api",
    )

# concurrency_count=5, max_size=20
# max_size=36, concurrency_count=14
block.queue(concurrency_count=5, max_size=20).launch(debug=True)
