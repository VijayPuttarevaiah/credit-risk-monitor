"""Monthly risk commentary, optionally drafted by Claude and always fact-checked.

The model only writes prose around numbers this pipeline already computed.
Every number in its draft is checked against those facts; if any number
does not match, the draft is rejected and the deterministic template is
used instead. The model never changes a figure or a decision.
"""

from __future__ import annotations

import json
import re
import sys

MODEL = "claude-opus-5-5"
NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
TOLERANCE = 0.051  # allows a figure to be rounded to one decimal place

SYSTEM_PROMPT = (
    "You write the monthly commentary for a credit card portfolio risk report. "
    "Use only the numbers in the facts JSON, quoted exactly as given or rounded to one "
    "decimal place. Do not compute new figures, do not add percentages that are not in "
    "the facts, and do not give lending advice. Write 4 to 6 plain sentences for a "
    "credit risk manager: model quality, where risk is concentrated, delinquency, and "
    "what the stress test shows."
)


def template(facts: dict) -> str:
    g = facts
    return (
        f"The scorecard ranks risk with a Gini of {g['gini']} and a KS of {g['ks']} on the holdout sample. "
        f"Grades D and E hold {g['share_d_e_pct']}% of accounts but {g['el_share_d_e_pct']}% of expected loss. "
        f"Accounts already 2 or more months late default next month at {g['default_rate_2plus_late_pct']}%, "
        f"against {g['default_rate_current_pct']}% for current accounts. "
        f"Portfolio expected loss is NT${g['expected_loss_baseline']:,} at baseline and rises "
        f"{g['severe_stress_increase_pct']}% under the severe scenario, which moves the number of "
        f"grade D and E accounts from {g['accounts_in_d_e_baseline']:,} to {g['accounts_in_d_e_severe']:,}. "
        f"Score stability between development and holdout samples is a PSI of {g['psi_dev_vs_holdout']}."
    )


def _numbers(text: str) -> list[float]:
    values = []
    for match in NUMBER_RE.findall(text):
        cleaned = match.replace(",", "").rstrip(".")
        if cleaned and cleaned not in {"-"}:
            values.append(float(cleaned))
    return values


def verify(text: str, facts: dict) -> list[float]:
    """Return every number in text that cannot be traced to a fact (empty list = passes)."""
    allowed = [float(v) for v in facts.values() if isinstance(v, (int, float))]
    allowed += [1, 2, 3, 4, 5, 6]  # small counts used in plain wording ("2 or more months")
    return [n for n in _numbers(text) if not any(abs(n - a) <= TOLERANCE for a in allowed)]


def _skip(reason: str) -> None:
    print(f"AI commentary skipped: {reason}. Using the template.", file=sys.stderr)


def draft_with_claude(facts: dict) -> str | None:
    """Ask Claude for a draft. Returns None if the SDK or credentials are unavailable."""
    try:
        import anthropic
    except ImportError:
        return None
    try:
        client = anthropic.Anthropic()
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": json.dumps(facts, indent=2)}],
        )
    except TypeError:
        # The SDK raises TypeError when no API key, auth token or profile is configured.
        _skip("no Anthropic credentials found")
        return None
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError):
        _skip("Anthropic credentials were rejected")
        return None
    except anthropic.APIConnectionError:
        _skip("could not reach the Anthropic API")
        return None
    except anthropic.APIStatusError as exc:
        _skip(f"Anthropic API returned {exc.status_code}")
        return None
    if response.stop_reason == "refusal":
        _skip("the model declined the request")
        return None
    return "".join(b.text for b in response.content if b.type == "text").strip() or None


def generate(facts: dict, use_ai: bool = True) -> tuple[str, str]:
    """Return (commentary, source) where source is 'claude' or 'template'."""
    if use_ai:
        draft = draft_with_claude(facts)
        if draft and not verify(draft, facts):
            return draft, "claude"
    return template(facts), "template"
