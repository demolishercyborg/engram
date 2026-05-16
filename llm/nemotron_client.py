"""
Nemotron local inference client.
Loads nvidia/NVIDIA-Nemotron-Nano-9B-v2 from a local snapshot_download path.
Compatible with NVIDIA GX-10 (Grace-Blackwell, CUDA, bfloat16).
"""

import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_instance = None   # module-level singleton — load once


class NemotronClient:
    def __init__(self, model_path: str):
        print(f"[Nemotron] Loading tokenizer from {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[Nemotron] Loading model (bfloat16, device_map=auto) ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"[Nemotron] Ready on {next(self.model.parameters()).device}.")

    def __call__(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        prompt = self._build_prompt(messages, tools)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens   = output_ids[0][inputs["input_ids"].shape[1]:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        tool_calls = _parse_tool_calls(response_text)
        return {
            "message": {
                "content":    "" if tool_calls else response_text,
                "tool_calls": tool_calls,
            }
        }

    def _build_prompt(self, messages: list[dict], tools: list[dict] | None) -> str:
        # Try the tokenizer's built-in chat template with tool support first.
        # Nemotron-Nano-9B-v2 is Qwen-2.5-based and supports this natively.
        try:
            if tools:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            pass

        # Manual fallback: inject tool descriptions into a system message.
        msgs = list(messages)
        if tools:
            tool_desc = json.dumps(
                [t["function"] for t in tools if "function" in t],
                indent=2,
            )
            header = (
                "You have access to the following tools. "
                "To call one, output ONLY a JSON block in the format:\n"
                '<tool_call>\n{"name": "<tool_name>", "arguments": {<args>}}\n</tool_call>\n\n'
                f"Tools:\n{tool_desc}"
            )
            if msgs and msgs[0]["role"] == "system":
                msgs[0] = {"role": "system", "content": msgs[0]["content"] + "\n\n" + header}
            else:
                msgs.insert(0, {"role": "system", "content": header})

        parts = []
        for m in msgs:
            role    = m["role"].capitalize()
            content = m["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>Assistant\n")
        return "\n".join(parts)


def _parse_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from Nemotron/Qwen-style <tool_call>...</tool_call> output."""
    patterns = [
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>",   # Qwen / Nemotron standard
        r"```(?:json)?\s*(\{[^`]*\"name\"\s*:[^`]*\})\s*```",  # markdown code block
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if not matches:
            continue
        result = []
        for m in matches:
            try:
                data = json.loads(m)
                args = data.get("arguments", data.get("parameters", {}))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                result.append({
                    "function": {
                        "name":      data.get("name", ""),
                        "arguments": args,
                    }
                })
            except (json.JSONDecodeError, KeyError):
                pass
        if result:
            return result
    return []


def make_llm_fn(model_path: str):
    """Returns a stateless callable compatible with EngramAgent.chat()."""
    global _instance
    if _instance is None:
        _instance = NemotronClient(model_path)

    def llm_fn(messages: list[dict], tools: list[dict] | None = None) -> dict:
        return _instance(messages, tools)

    return llm_fn
