# gstar/policy.py
from typing import Iterable, List

def _fallback_from(options: List[str]) -> str:
    for fb in ("look", "inventory"):
        if fb in options:
            return fb
    return options[0]

def _llm_pick_verbatim(llm, obs: str, options: List[str], goal: str = "") -> str:
    """Use an OpenAI-compatible LLM to return EXACTLY one action from options (English-only prompt)."""
    system = (
        "You are a household task agent in a text-based environment. "
        "ALWAYS output exactly one action from OPTIONS. No explanations."
    )
    user = (
        (f"Goal:\n{goal}\n\n" if goal else "") +
        "Observation:\n" + obs + "\n\n" +
        "OPTIONS (candidate actions, one per line):\n" + "\n".join(options) + "\n\n" +
        "RULES:\n"
        "1) Output EXACTLY one line.\n"
        "2) The output MUST be copied verbatim from OPTIONS.\n"
        "3) Do NOT add extra text.\n"
        "OUTPUT:"
    )
    resp = llm.chat.completions.create(
        model=getattr(llm, "model", None) or "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0,
        max_tokens=8,
        stop=["\n"],
    )
    return (resp.choices[0].message.content or "").strip()

def select_action(obs: str, admissible: Iterable[str], llm, goal: str = "") -> str:
    """
    Single entry to choose the next action for ALFWorld.
    ✅ Keep this function as the ONLY place to decide actions.
    🔁 Replace the LLM block below with YOUR GStar/A* policy when ready.
    """
    options = sorted(set(admissible)) or ["look", "inventory"]

    # === YOUR POLICY HOOK (sample) ============================================
    # If you already have your own planner, call it here and ensure validity:
    #   from gstar.my_planner import next_action
    #   cand = next_action(obs=obs, goal=goal, options=options)
    #   return cand if cand in options else _fallback_from(options)
    # ==========================================================================

    # Placeholder: LLM picks one verbatim option (safe, deterministic)
    action = _llm_pick_verbatim(llm, obs, options, goal)

    # Final safety check: must be inside options
    if action not in options:
        return _fallback_from(options)
    return action
