"""Prompt construction, split so the expensive parts can be KV-cached.

Layout of every request (causal attention => static things first):

    [ STATIC ]  chat header + system prompt + few-shot examples + "STATE:\\n"   (cached once per engine)
    [ STATE  ]  the request payload                                              (one pass per request)
    [ SUFFIX ]  one typed question + end-of-user-turn + assistant header + cue   (short, one per question)

The three parts are tokenized SEPARATELY and concatenated as ids, so the cached prefix ids are
always identical no matter which question follows.
"""
from __future__ import annotations

import json
from typing import Any, List, Optional, Tuple

from .schema import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion
from .symbols import CHOICE_SYMBOLS_255

LETTERS = "".join(CHOICE_SYMBOLS_255)
STATE_MARK = "<<STATE>>"
QUESTION_MARK = "<<QUESTION>>"

SYSTEM_PROMPT_GENERAL = (
    "You are a fast, precise decision engine. You receive a STATE and one typed QUESTION. "
    "Reply with the single label requested (an option letter, a scale number, or Yes/No). "
    "Never explain."
)
SYSTEM_PROMPT = SYSTEM_PROMPT_GENERAL

SYSTEM_PROMPT_CHOICE = (
    "You are a fast, precise categorical decision engine. You receive a STATE and a choice QUESTION. "
    "Compare the state carefully against all provided options and choose the single best-fitting option. "
    "Reply with the option letter only (e.g. A, B, C). Never explain."
)

SYSTEM_PROMPT_SCORE = (
    "You are a fast, precise ordinal evaluation engine. You receive a STATE and a rubric scale QUESTION. "
    "Evaluate the state objectively against each level criteria. Do not exaggerate or overestimate severity; "
    "only select higher scale numbers when the state explicitly satisfies the higher criteria. "
    "Reply with the scale number only (e.g. 0, 1, 2). Never explain."
)

SYSTEM_PROMPT_NOUL = (
    "You are a fast, precise boolean decision engine. You receive a STATE and a verification QUESTION. "
    "Determine whether the condition is strictly satisfied. Do not assume or guess 'Yes' unless the facts clearly confirm it. "
    "Reply with Yes or No only. Never explain."
)


def render_state(state: Any) -> str:
    if isinstance(state, str):
        return state.strip()
    if isinstance(state, dict):
        return "\n".join(f"{k}: {v}" for k, v in state.items())
    return json.dumps(state, ensure_ascii=False, indent=2)


def render_question(q: Question, option_order: Optional[List[str]] = None) -> Tuple[str, List[str]]:
    """Return (question_text, label_strings). For choice questions `option_order` is the
    display order of option keys; label i (letter) refers to option_order[i]."""
    if isinstance(q, ChoiceQuestion):
        order = option_order or list(q.options)
        lines = [f"QUESTION (choose exactly one option): {q.instructions}", "Options:"]
        for letter, key in zip(LETTERS, order):
            desc = q.options[key]
            lines.append(f"{letter}. {key}" + (f" - {desc}" if desc else ""))
        lines.append("Reply with the option letter only.")
        return "\n".join(lines) + "\n", list(LETTERS[: len(order)])

    if isinstance(q, ScoreQuestion):
        lines = [f"QUESTION (rate on a scale): {q.instructions}", "Scale:"]
        lines += [f"{i} = {desc}" for i, desc in enumerate(q.levels)]
        lines.append("Reply with the scale number only.")
        return "\n".join(lines) + "\n", [str(i) for i in range(len(q.levels))]

    if isinstance(q, NoulQuestion):
        text = f"QUESTION (yes/no): {q.instructions}\nReply Yes or No only.\n"
        return text, ["Yes", "No"]

    raise TypeError(f"Unsupported question: {q!r}")


# --- few-shot examples (generic on purpose: they must not overlap with the test set) ---------
def _fewshot_block() -> str:
    from .schema import parse_question

    examples = [
        ("Customer message: The lid on my parcel arrived cracked.",
         parse_question("x", {"type": "choice", "instructions": "Which team should handle this message?",
                              "options": {"billing": "payments and invoices",
                                          "shipping": "delivery damage and lost parcels",
                                          "product_info": "questions about product features"}}),
         ["billing", "shipping", "product_info"], "B"),
        ("Customer message: Does the X200 support Bluetooth 5?",
         parse_question("x", {"type": "choice", "instructions": "Which team should handle this message?",
                              "options": {"product_info": "questions about product features",
                                          "billing": "payments and invoices",
                                          "shipping": "delivery damage and lost parcels"}}),
         ["product_info", "billing", "shipping"], "A"),
        ("Message: Just wondering if you sell gift cards.",
         parse_question("x", {"type": "score", "instructions": "How urgent is this message?",
                              "levels": ["low", "medium", "high"]}), None, "0"),
        ("Message: My site is down and customers cannot check out.",
         parse_question("x", {"type": "score", "instructions": "How urgent is this message?",
                              "levels": ["low", "medium", "high"]}), None, "2"),
        ("Message: I love the new update, thank you!",
         parse_question("x", {"type": "noul", "instructions": "Is the writer unhappy?"}), None, "No"),
        ("Message: This is the third time it broke. I want my money back.",
         parse_question("x", {"type": "noul", "instructions": "Is the writer unhappy?"}), None, "Yes"),
    ]
    parts = ["Examples:"]
    for state, q, order, answer in examples:
        qtext, _ = render_question(q, order)
        parts.append(f"STATE:\n{state}\n\n{qtext}Answer: {answer}")
    return "\n\n".join(parts)


def _fewshot_score_block() -> str:
    """Dedicated 5-shot calibration examples for rubric score questions."""
    from .schema import parse_question

    examples = [
        # Example 1: Level 0 on 3-point scale (cosmetic / style nitpick)
        (
            "Code Review Comment: Variable 'user_id' in helper function is missing a docstring and has trailing spaces.",
            parse_question("x", {
                "type": "score",
                "instructions": "Rate the defect severity of this issue.",
                "levels": [
                    "cosmetic: formatting, style guidelines, or docstring suggestion",
                    "functional: incorrect calculation, unhandled edge case, or logic flaw",
                    "critical: security exploit, credential exposure, or crash",
                ],
            }),
            None,
            "0",
        ),
        # Example 2: Level 1 on 3-point scale (intermediate / moderate degradation)
        (
            "IoT Diagnostic Log: Sensor active for 14 months. Current battery capacity at 72% of factory rating. Normal telemetry transmissions continuing without brownouts.",
            parse_question("x", {
                "type": "score",
                "instructions": "Rate the battery degradation level.",
                "levels": [
                    "healthy: battery capacity above 85% with normal discharge",
                    "moderate: capacity between 60% and 85%, noticeable runtime decline",
                    "severe: capacity below 60%, swelling, or intermittent brownouts",
                ],
            }),
            None,
            "1",
        ),
        # Example 3: Level 2 on 3-point scale (hazardous / high severity)
        (
            "Acoustic Monitor: Sustained industrial turbine siren sounding at 96 decibels directly adjacent to the inspection platform.",
            parse_question("x", {
                "type": "score",
                "instructions": "Rate the acoustic hazard level.",
                "levels": [
                    "quiet: ambient room or library levels (<50 dB)",
                    "elevated: busy office or conversation levels (50-75 dB)",
                    "hazardous: high decibel machinery or alarms requiring immediate hearing protection (>85 dB)",
                ],
            }),
            None,
            "2",
        ),
        # Example 4: Level 1 on 4-point scale (low / intermediate drift)
        (
            "Data Pipeline Metric: Daily numerical feature distribution shifted slightly with Wasserstein distance 0.04. Downstream predictions remain within tolerance.",
            parse_question("x", {
                "type": "score",
                "instructions": "Rate the severity of feature distribution drift.",
                "levels": [
                    "none: feature distribution fully identical to baseline",
                    "low: minor statistical variance, continue monitoring without pipeline intervention",
                    "substantial: pronounced distribution skew, model retraining recommended",
                    "breaking: missing schema columns, incompatible data types, or pipeline failure",
                ],
            }),
            None,
            "1",
        ),
        # Example 5: Level 0 on 4-point scale (baseline pristine condition)
        (
            "Warehouse Intake: Returned electronics item received in original factory packaging with unbroken security tape. Item has never been powered on.",
            parse_question("x", {
                "type": "score",
                "instructions": "Rate the physical wear and tear on this returned merchandise.",
                "levels": [
                    "pristine: unopened manufacturer packaging, brand new condition",
                    "open box: packaging opened but item shows no signs of handling or wear",
                    "visible wear: surface scratches or cosmetic blemishes from prior usage",
                    "broken: structural damage or non-operational",
                ],
            }),
            None,
            "0",
        ),
    ]
    parts = ["Examples:"]
    for state, q, order, answer in examples:
        qtext, _ = render_question(q, order)
        parts.append(f"STATE:\n{state}\n\n{qtext}Answer: {answer}")
    return "\n\n".join(parts)


def _fewshot_choice_block() -> str:
    """Dedicated 3-shot examples for categorical choice questions using distinct
    option notation (1), (2), (3) disjoint from CHOICE_SYMBOLS_255 to eliminate token frequency bias.
    """
    examples = [
        # Example 1: Language Classification
        (
            "Text Snippet: Bonjour a tous, nous vous remercions sincerement d'avoir assiste a notre conference annuelle.",
            "QUESTION (choose exactly one option): Identify the primary language of the text.\n"
            "Options:\n"
            "(1) en - English\n"
            "(2) fr - French\n"
            "(3) de - German\n"
            "(4) es - Spanish\n"
            "Reply with the option label only.\n",
            "(2)",
        ),
        # Example 2: Meeting Scheduling Category
        (
            "Meeting Description: Weekly engineering alignment to review backlog items, resolve dependency blockers, and assign sprint tasks.",
            "QUESTION (choose exactly one option): Which category best fits this meeting?\n"
            "Options:\n"
            "(1) social - informal team building or celebration\n"
            "(2) standup - regular team sync and operational progress tracking\n"
            "(3) incident - live production emergency response\n"
            "Reply with the option label only.\n",
            "(2)",
        ),
        # Example 3: Transportation Mode
        (
            "Route Spec: Transporting bulk cargo 3,200 miles across the Pacific Ocean from Yokohama to San Francisco.",
            "QUESTION (choose exactly one option): Which primary transport mode is required?\n"
            "Options:\n"
            "(1) truck - highway freight vehicle\n"
            "(2) rail - regional train freight\n"
            "(3) maritime - ocean cargo ship\n"
            "(4) pedestrian - walking courier\n"
            "Reply with the option label only.\n",
            "(3)",
        ),
    ]
    parts = ["Examples:"]
    for state, qtext, answer in examples:
        parts.append(f"STATE:\n{state}\n\n{qtext}Answer: {answer}")
    return "\n\n".join(parts)


def _fewshot_noul_block() -> str:
    """Dedicated 4-shot examples for boolean noul questions (balanced 2 Yes / 2 No)."""
    from .schema import parse_question

    examples = [
        # Example 1: Condition not met (No)
        (
            "Recipe Ingredients: Organic oat flour, brown sugar, almond milk, baking powder, and vanilla extract.",
            parse_question("x", {
                "type": "noul",
                "instructions": "Does this recipe contain dairy ingredients?",
            }),
            None,
            "No",
        ),
        # Example 2: Condition satisfied (Yes)
        (
            "Server Specification: High-performance node with 128 GB DDR5 ECC memory and dual 25 Gbps network interfaces.",
            parse_question("x", {
                "type": "noul",
                "instructions": "Does this server have at least 64 GB of memory?",
            }),
            None,
            "Yes",
        ),
        # Example 3: Prerequisite not met (No)
        (
            "Candidate Profile: Developer with 2 years of professional Python programming and basic Docker familiarity.",
            parse_question("x", {
                "type": "noul",
                "instructions": "Does this candidate have 5 or more years of professional experience?",
            }),
            None,
            "No",
        ),
        # Example 4: Condition satisfied (Yes)
        (
            "Vehicle Report: Front windshield has an 8-inch spiderweb fracture directly obstructing the driver's forward view.",
            parse_question("x", {
                "type": "noul",
                "instructions": "Does the vehicle have glass damage that impairs driver visibility?",
            }),
            None,
            "Yes",
        ),
    ]
    parts = ["Examples:"]
    for state, q, order, answer in examples:
        qtext, _ = render_question(q, order)
        parts.append(f"STATE:\n{state}\n\n{qtext}Answer: {answer}")
    return "\n\n".join(parts)


class PromptBuilder:
    def __init__(
        self,
        tokenizer,
        use_chat_template: Optional[bool] = None,
        few_shot: bool = True,
        answer_cue: str = "Answer:",
        enable_thinking: bool = False,
        qtype: Optional[str] = None,
    ):
        self.tok = tokenizer
        self.use_chat_template = (
            bool(getattr(tokenizer, "chat_template", None)) if use_chat_template is None else use_chat_template
        )
        self.answer_cue = answer_cue
        self.qtype = qtype

        if qtype == "choice":
            sys_text = SYSTEM_PROMPT_CHOICE
            fs_block = _fewshot_choice_block() if few_shot else ""
        elif qtype == "score":
            sys_text = SYSTEM_PROMPT_SCORE
            fs_block = _fewshot_score_block() if few_shot else ""
        elif qtype == "noul":
            sys_text = SYSTEM_PROMPT_NOUL
            fs_block = _fewshot_noul_block() if few_shot else ""
        else:
            sys_text = SYSTEM_PROMPT_GENERAL
            fs_block = _fewshot_block() if few_shot else ""

        system = sys_text + (("\n\n" + fs_block) if fs_block else "")
        text = self._render_template(system, enable_thinking)
        static_text, rest = text.split(STATE_MARK, 1)
        self._between, self._tail = rest.split(QUESTION_MARK, 1)
        self._static_text = static_text

    def _render_template(self, system: str, enable_thinking: bool) -> str:
        user = f"STATE:\n{STATE_MARK}\n\n{QUESTION_MARK}"
        if self.use_chat_template:
            kw = dict(tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
            try:
                return self.tok.apply_chat_template(
                    [{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)
            except Exception:  # templates without a system role
                return self.tok.apply_chat_template(
                    [{"role": "user", "content": system + "\n\n" + user}], **kw)
        bos = getattr(self.tok, "bos_token", None) or ""
        return f"{bos}{system}\n\nSTATE:\n{STATE_MARK}\n\n{QUESTION_MARK}"

    def _enc(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def static_ids(self) -> List[int]:
        return self._enc(self._static_text)

    def state_ids(self, state: Any) -> List[int]:
        return self._enc(render_state(state) + self._between)

    def suffix_ids(self, question_text: str) -> List[int]:
        return self._enc(question_text + self._tail + self.answer_cue)
