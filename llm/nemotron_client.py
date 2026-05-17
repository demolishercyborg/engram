from __future__ import annotations
"""
Nemotron local inference client.
Loads nvidia/NVIDIA-Nemotron-Nano-9B-v2 from a local snapshot_download path.
Compatible with NVIDIA GX-10 (Grace-Blackwell, CUDA, bfloat16).
"""

import json
import re
import time
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

        print("[Nemotron] Loading model (dtype=auto, device_map=auto) ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
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
        if tool_calls:
            # Strip the call markup so the assistant's content carries only
            # any free-form reasoning the model emitted alongside the call.
            content = _strip_tool_call_markup(response_text)
            # Stamp ids so downstream tool result messages can reference them.
            base = int(time.time() * 1000)
            for i, tc in enumerate(tool_calls):
                tc.setdefault("id", f"call_{base}_{i}")
                tc.setdefault("type", "function")
        else:
            content = response_text

        return {
            "message": {
                "content":    content,
                "tool_calls": tool_calls,
            }
        }

    def _build_prompt(self, messages: list[dict], tools: list[dict] | None) -> str:
        # Try the tokenizer's built-in chat template with tool support first.
        # Nemotron-Nano-9B-v2 is a Mamba-Transformer hybrid (Nemotron-H);
        # tool calling via apply_chat_template works if the tokenizer config supports it.
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

        # Nemotron-H uses <SPECIAL_10> System / <SPECIAL_11> User/Assistant / <SPECIAL_12> EOS
        _role_token = {"system": "<SPECIAL_10>", "user": "<SPECIAL_11>", "assistant": "<SPECIAL_11>"}
        parts = []
        for m in msgs:
            role    = m["role"].lower()
            token   = _role_token.get(role, "<SPECIAL_11>")
            label   = role.capitalize()
            content = m["content"]
            parts.append(f"{token}{label}\n{content}\n<SPECIAL_12>")
        parts.append("<SPECIAL_11>Assistant\n")
        return "\n".join(parts)


_TOOL_CALL_MARKUP_PATTERNS = [
    re.compile(r"<TOOLCALL>.*?</TOOLCALL>", re.DOTALL),
    re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL),
    re.compile(r"```(?:json)?\s*\{[^`]*\"name\"\s*:[^`]*\}\s*```", re.DOTALL),
]


def _strip_tool_call_markup(text: str) -> str:
    for pat in _TOOL_CALL_MARKUP_PATTERNS:
        text = pat.sub("", text)
    return text.strip()


def _parse_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from model output, supporting multiple formats."""

    def _normalise(data: dict) -> dict:
        args = data.get("arguments", data.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        return {"function": {"name": data.get("name", ""), "arguments": args}}

    # Nemotron format: <TOOLCALL>[{...}, ...]</TOOLCALL>  (JSON array)
    m = re.search(r"<TOOLCALL>\s*(\[.*?\])\s*</TOOLCALL>", text, re.DOTALL)
    if m:
        try:
            calls = json.loads(m.group(1))
            result = [_normalise(c) for c in calls if isinstance(c, dict)]
            if result:
                return result
        except (json.JSONDecodeError, TypeError):
            pass

    # Qwen format: <tool_call>{...}</tool_call>  (one object per tag)
    single_matches = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if single_matches:
        result = []
        for raw in single_matches:
            try:
                result.append(_normalise(json.loads(raw)))
            except (json.JSONDecodeError, KeyError):
                pass
        if result:
            return result

    # Markdown JSON code block fallback
    block_matches = re.findall(r"```(?:json)?\s*(\{[^`]*\"name\"\s*:[^`]*\})\s*```", text, re.DOTALL)
    if block_matches:
        result = []
        for raw in block_matches:
            try:
                result.append(_normalise(json.loads(raw)))
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
