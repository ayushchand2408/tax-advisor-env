"""
TaxAdvisorEnv — OpenEnv compliant environment for Meta Hackathon
Real-world task: Help an AI agent prepare and file taxes.
"""

from __future__ import annotations
from typing import Any
from pydantic import BaseModel


# ─── Typed Models ────────────────────────────────────────────────────────────

class TaxAction(BaseModel):
    tool_name: str          # e.g. "get_income_data", "compute_taxes", "submit_form"
    arguments: dict         # tool-specific arguments


class TaxObservation(BaseModel):
    data: dict              # result data from the tool call
    message: str            # human-readable description
    success: bool           # whether the action succeeded


class TaxState(BaseModel):
    task_id: int            # 0=easy, 1=medium, 2=hard
    fields_filled: int      # how many form fields have been correctly filled
    total_fields: int       # total fields required for this task
    deductions_found: int   # number of valid deductions identified (tasks 1 & 2)
    total_deductions: int   # expected number of deductions
    submitted: bool         # whether the final form was submitted
    last_action: str        # last tool called
    steps_taken: int        # total steps used so far


# ─── Synthetic Data ───────────────────────────────────────────────────────────

SYNTHETIC_PROFILES = [
    {
        "name": "Alice Kumar",
        "income": 75000,
        "filing_status": "single",
        "receipts": [
            {"category": "home_office",    "amount": 1200, "deductible": True},
            {"category": "personal_food",  "amount": 3000, "deductible": False},
            {"category": "education",      "amount": 800,  "deductible": True},
            {"category": "health_expense", "amount": 600,  "deductible": True},
            {"category": "luxury_vacation","amount": 4000, "deductible": False},
        ],
        "bank_data": {"wages": 75000, "interest_income": 200},
        "correct_tax": 12000,   # simplified flat bracket for demo
    },
    {
        "name": "Bob Sharma",
        "income": 120000,
        "filing_status": "married",
        "receipts": [
            {"category": "charitable_donation", "amount": 2000, "deductible": True},
            {"category": "mortgage_interest",   "amount": 9000, "deductible": True},
            {"category": "personal_clothing",   "amount": 1500, "deductible": False},
            {"category": "business_travel",     "amount": 3200, "deductible": True},
            {"category": "gym_membership",      "amount": 600,  "deductible": False},
        ],
        "bank_data": {"wages": 120000, "interest_income": 500},
        "correct_tax": 22000,
    },
]

TAX_BRACKETS = [
    (10275,  0.10),
    (41775,  0.12),
    (89075,  0.22),
    (170050, 0.24),
    (215950, 0.32),
    (539900, 0.35),
    (float("inf"), 0.37),
]


def compute_tax(income: float) -> float:
    """Simple US-style progressive tax calculation."""
    tax = 0.0
    prev = 0.0
    for bracket, rate in TAX_BRACKETS:
        if income <= bracket:
            tax += (income - prev) * rate
            break
        tax += (bracket - prev) * rate
        prev = bracket
    return round(tax, 2)


# ─── Environment ─────────────────────────────────────────────────────────────

class TaxAdvisorEnv:
    """
    OpenEnv-compliant Tax Advisor Environment.

    Three tasks:
      Task 0 (easy)   — Calculate tax from a simple income statement.
      Task 1 (medium) — Identify deductible expenses from raw receipts.
      Task 2 (hard)   — Full filing: collect data, compute tax, submit form.

    step(action)  → (TaxObservation, reward: float, done: bool, info: dict)
    reset()       → TaxObservation
    state()       → TaxState
    """

    AVAILABLE_TOOLS = [
        "get_income_data",
        "get_receipts",
        "classify_expense",
        "search_tax_code",
        "compute_taxes",
        "fill_form_field",
        "submit_form",
    ]

    def __init__(self, task_id: int = 0, profile_index: int = 0):
        assert task_id in (0, 1, 2), "task_id must be 0, 1, or 2"
        self.task_id = task_id
        self.profile = SYNTHETIC_PROFILES[profile_index % len(SYNTHETIC_PROFILES)]
        self._state: TaxState = None
        self._form_fields: dict = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> TaxObservation:
        """Reset environment to initial state. Returns opening observation."""
        total_deductions = sum(
            1 for r in self.profile["receipts"] if r["deductible"]
        )
        self._form_fields = {}
        self._state = TaxState(
            task_id=self.task_id,
            fields_filled=0,
            total_fields=self._get_total_fields(),
            deductions_found=0,
            total_deductions=total_deductions,
            submitted=False,
            last_action="reset",
            steps_taken=0,
        )
        task_descriptions = {
            0: "Task 0 (Easy): You have a taxpayer's income data. Calculate the correct tax owed.",
            1: "Task 1 (Medium): You have a list of expense receipts. Classify each as deductible or not.",
            2: "Task 2 (Hard): Complete a full tax filing — gather data, find deductions, compute tax, and submit.",
        }
        return TaxObservation(
            data={"available_tools": self.AVAILABLE_TOOLS, "task_id": self.task_id},
            message=task_descriptions[self.task_id],
            success=True,
        )

    def step(self, action: TaxAction) -> tuple[TaxObservation, float, bool, dict]:
        """
        Execute one action.
        Returns: (observation, reward, done, info)
        """
        self._state.steps_taken += 1
        self._state.last_action = action.tool_name

        if action.tool_name not in self.AVAILABLE_TOOLS:
            obs = TaxObservation(
                data={},
                message=f"Unknown tool '{action.tool_name}'. Available: {self.AVAILABLE_TOOLS}",
                success=False,
            )
            return obs, -0.05, False, {"error": "unknown_tool"}

        # Dispatch to handler
        handler = getattr(self, f"_tool_{action.tool_name}", None)
        obs, reward = handler(action.arguments)

        done = self._check_done()
        if done and not obs.data.get("already_counted_bonus"):
            reward += self._completion_bonus()

        return obs, round(reward, 4), done, {"state": self._state.model_dump()}

    def state(self) -> TaxState:
        """Return current environment state."""
        return self._state

    # ── Tool Handlers ─────────────────────────────────────────────────────────

    def _tool_get_income_data(self, args: dict) -> tuple[TaxObservation, float]:
        data = {
            "name": self.profile["name"],
            "wages": self.profile["bank_data"]["wages"],
            "interest_income": self.profile["bank_data"]["interest_income"],
            "filing_status": self.profile["filing_status"],
        }
        return TaxObservation(data=data, message="Income data retrieved.", success=True), 0.05

    def _tool_get_receipts(self, args: dict) -> tuple[TaxObservation, float]:
        receipts = [
            {"category": r["category"], "amount": r["amount"]}
            for r in self.profile["receipts"]
        ]
        return TaxObservation(data={"receipts": receipts}, message="Receipts retrieved.", success=True), 0.05

    def _tool_classify_expense(self, args: dict) -> tuple[TaxObservation, float]:
        category = args.get("category", "")
        is_deductible = args.get("is_deductible", None)

        match = next(
            (r for r in self.profile["receipts"] if r["category"] == category), None
        )
        if not match:
            return TaxObservation(
                data={}, message=f"Category '{category}' not found.", success=False
            ), -0.05

        correct = match["deductible"]
        if is_deductible == correct:
            self._state.deductions_found += 1
            reward = 0.2
            msg = f"Correct! '{category}' is {'deductible' if correct else 'not deductible'}."
        else:
            reward = -0.1
            msg = f"Wrong. '{category}' is {'deductible' if correct else 'not deductible'}."

        return TaxObservation(
            data={"category": category, "correct": correct},
            message=msg,
            success=(is_deductible == correct),
        ), reward

    def _tool_search_tax_code(self, args: dict) -> tuple[TaxObservation, float]:
        query = args.get("query", "").lower()
        rules = {
            "home_office": "Home office deduction allowed if used exclusively for business (IRC §280A).",
            "education": "Education expenses deductible if related to current job (IRC §162).",
            "charitable": "Charitable donations deductible up to 60% of AGI (IRC §170).",
            "mortgage": "Mortgage interest deductible on first $750,000 of debt (IRC §163).",
            "health": "Medical expenses deductible if >7.5% of AGI (IRC §213).",
            "business_travel": "Ordinary and necessary business travel is deductible (IRC §162).",
        }
        result = {k: v for k, v in rules.items() if k in query or query in k}
        if not result:
            result = rules  # return all rules if no match
        return TaxObservation(
            data={"tax_rules": result},
            message="Tax code search complete.",
            success=True,
        ), 0.05

    def _tool_compute_taxes(self, args: dict) -> tuple[TaxObservation, float]:
        income = args.get("income", 0)
        deductions = args.get("deductions", 0)
        taxable_income = max(0, income - deductions)
        computed = compute_tax(taxable_income)
        expected = self.profile["correct_tax"]
        error_pct = abs(computed - expected) / max(expected, 1)

        if error_pct <= 0.05:   # within 5% tolerance
            reward = 0.3
            msg = f"Tax computed: ${computed:,.2f}. Correct!"
            self._state.fields_filled = min(
                self._state.fields_filled + 1, self._state.total_fields
            )
        else:
            reward = -0.05
            msg = f"Tax computed: ${computed:,.2f}. Check your inputs."

        return TaxObservation(
            data={"computed_tax": computed, "taxable_income": taxable_income},
            message=msg,
            success=(error_pct <= 0.05),
        ), reward

    def _tool_fill_form_field(self, args: dict) -> tuple[TaxObservation, float]:
        field = args.get("field", "")
        value = args.get("value")
        valid_fields = {
            "taxpayer_name": self.profile["name"],
            "filing_status": self.profile["filing_status"],
            "total_income": self.profile["income"],
            "total_deductions": sum(
                r["amount"] for r in self.profile["receipts"] if r["deductible"]
            ),
            "tax_owed": self.profile["correct_tax"],
        }
        if field not in valid_fields:
            return TaxObservation(
                data={}, message=f"Unknown field '{field}'.", success=False
            ), -0.05

        expected = valid_fields[field]
        # Flexible matching: strings case-insensitive, numbers within 5%
        if isinstance(expected, str):
            correct = str(value).lower() == str(expected).lower()
        else:
            try:
                correct = abs(float(value) - float(expected)) / max(float(expected), 1) <= 0.05
            except Exception:
                correct = False

        if correct and field not in self._form_fields:
            self._form_fields[field] = value
            self._state.fields_filled += 1
            reward = 0.2
            msg = f"Field '{field}' filled correctly."
        elif correct:
            reward = 0.0
            msg = f"Field '{field}' already filled."
        else:
            reward = -0.1
            msg = f"Field '{field}' incorrect. Expected type: {type(expected).__name__}."

        return TaxObservation(
            data={"field": field, "accepted": correct},
            message=msg,
            success=correct,
        ), reward

    def _tool_submit_form(self, args: dict) -> tuple[TaxObservation, float]:
        if self._state.fields_filled < self._state.total_fields:
            missing = self._state.total_fields - self._state.fields_filled
            return TaxObservation(
                data={"fields_missing": missing},
                message=f"Cannot submit: {missing} fields still incomplete.",
                success=False,
            ), -0.1

        self._state.submitted = True
        return TaxObservation(
            data={"status": "submitted", "fields_completed": self._state.fields_filled},
            message="Tax form submitted successfully!",
            success=True,
        ), 0.5

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_total_fields(self) -> int:
        return {0: 1, 1: len(self.profile["receipts"]), 2: 5}[self.task_id]

    def _check_done(self) -> bool:
        if self.task_id == 0:
            return self._state.fields_filled >= 1
        elif self.task_id == 1:
            return self._state.deductions_found >= self._state.total_deductions
        else:
            return self._state.submitted

    def _completion_bonus(self) -> float:
        """Dense + sparse reward: partial progress + full completion bonus."""
        progress = self._state.fields_filled / max(self._state.total_fields, 1)
        return round(1.0 * progress, 4)  # up to +1.0 for full completion


# ─── Graders (score 0.0–1.0) ─────────────────────────────────────────────────

def grade_task(env: TaxAdvisorEnv) -> float:
    """
    Returns a score strictly in (0.0, 1.0) — never exactly 0 or 1.
    Task 0: tax computed correctly.
    Task 1: fraction of deductions correctly classified.
    Task 2: fraction of form fields filled + submission bonus.
    """
    s = env.state()

    if s.task_id == 0:
        raw = 0.95 if s.fields_filled >= 1 else 0.05
    elif s.task_id == 1:
        fraction = s.deductions_found / max(s.total_deductions, 1)
        raw = fraction
    else:
        field_score = s.fields_filled / max(s.total_fields, 1)
        submit_bonus = 0.2 if s.submitted else 0.0
        raw = min(field_score * 0.8 + submit_bonus, 0.99)

    # Clamp strictly to (0.01, 0.99) — never exactly 0 or 1
    clamped = max(0.01, min(0.99, raw))
    return round(clamped, 4)