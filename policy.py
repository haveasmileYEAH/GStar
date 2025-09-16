# gstar/policy.py
from typing import Iterable, List

def _verbatim_from_options(llm, obs: str, options: List[str]) -> str:
    """
    Use the LLM to return EXACTLY one action string from `options`.
    English-only prompt to match ALFWorld's format.
    """
    system = "You are a household task agent in a text-based environment. ALWAYS output exactly one action from OPTIONS. No explanations."
    user = (
        "Observation:\n"
        f"{obs}\n\n"
        "OPTIONS (candidate actions, one per line):\n"
        + "\n".join(options)
        + "\n\n"
        "RULES:\n"
        "1) Output EXACTLY one line.\n"
        "2) The output MUST be copied verbatim from OPTIONS.\n"
        "3) Do NOT add extra text.\n"
        "OUTPUT:"
    )
    # OpenAI-compatible call via your llm wrapper/client
    resp = llm.chat.completions.create(
        model=llm.model,  # your wrapper can expose .model; if not, just pass a model str here.
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0,
        max_tokens=8,
        stop=["\n"]
    )
    return (resp.choices[0].message.content or "").strip()

def select_action(obs: str, admissible: Iterable[str], llm) -> str:
    """
    Single point to choose the next action for ALFWorld.
    Replace the body with your own GStar / A* policy when ready.
    - `obs`: current observation text (English)
    - `admissible`: iterable of admissible command strings (English)
    - `llm`: an OpenAI-compatible client already configured (Together/DashScope/etc.)
    """
    # Normalize options to a small, deterministic list
    opts = sorted(set(admissible))
    if not opts:
        return "look"  # safe fallback

    # TODO: Replace this block with your GStar/A* policy using the same `opts`
    # e.g. score each candidate and return the best; LLM below is a placeholder.
    action = _verbatim_from_options(llm, obs, opts)

    # Safety: ensure the action is valid; if not, fall back
    if action not in opts:
        # simple heuristic fallback
        for fallback in ("look", "inventory"):
            if fallback in opts:
                return fallback
        return opts[0]
    return action
