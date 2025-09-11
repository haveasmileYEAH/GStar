# llm_client.py
import os
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY",  "")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL",    "qwen-plus")

def chat(messages, max_tokens=512, temperature=0.2, model=None):
    """
    Unified chat() for both new & old openai clients.
    messages = [{"role":"user"/"assistant"/"system", "content":"..."}]
    """
    mdl = model or OPENAI_MODEL
    # Try new OpenAI SDK (>=1.0)
    try:
        from openai import OpenAI
        client = OpenAI(base_url=OPENAI_API_BASE, api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=mdl, messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        txt = resp.choices[0].message.content
        usage = getattr(resp, "usage", None)
        return txt, usage.__dict__ if usage else {}
    except Exception:
        pass

    # Fallback: old openai SDK (0.x style)
    try:
        import openai
        openai.api_base = OPENAI_API_BASE
        openai.api_key  = OPENAI_API_KEY
        resp = openai.ChatCompletion.create(
            model=mdl, messages=messages, max_tokens=max_tokens, temperature=temperature, n=1
        )
        txt = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
        return txt, usage
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")
