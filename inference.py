"""
inference.py — LLM-Powered Agent for TaxAdvisorEnv
Uses HuggingFace Inference API (free) via OpenAI-compatible client.

Setup:
    export HF_TOKEN=your_huggingface_token_here
    python inference.py

Get your free token at: https://huggingface.co/settings/tokens
"""

import os
import json
import time
from env import TaxAdvisorEnv, TaxAction, grade_task

# ─── API Setup ────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
MAX_STEPS    = 20

client = None
if API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        print(f"✅ LLM client initialized → {MODEL_NAME}")
    except Exception as e:
        print(f"⚠️  LLM setup failed: {e}")
else:
    print("⚠️  No HF_TOKEN found — falling back to rule-based agent")
    print("    Get a free token at https://huggingface.co/settings/tokens")
    print("    Then run: export HF_TOKEN=your_token")

# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a tax filing AI agent. Your job is to complete tax tasks
by calling tools one at a time.

Available tools:
- get_income_data       : {}
- get_receipts          : {}
- search_tax_code       : {"query": "<topic>"}
- classify_expense      : {"category": "<n>", "is_deductible": true or false}
- compute_taxes         : {"income": <number>, "deductions": <number>}
- fill_form_field       : {"field": "<n>", "value": <value>}
  Valid fields: taxpayer_name, filing_status, total_income, total_deductions, tax_owed
- submit_form           : {}

Rules:
- Always start by calling get_income_data and get_receipts to understand the situation
- For deductions: home_office, education, health_expense, charitable_donation,
  mortgage_interest, business_travel are deductible
- Personal expenses like food, clothing, vacations, gym are NOT deductible
- For Task 2 (hard): fill ALL form fields before calling submit_form

Respond with ONLY a JSON object — no explanation, no markdown, just raw JSON:
{"tool_name": "<tool>", "arguments": {}}"""

# ─── LLM Agent ────────────────────────────────────────────────────────────────

def llm_agent_step(messages: list):
    """Ask the LLM what action to take next. Returns (parsed, raw)."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=150,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        return parsed, raw
    except json.JSONDecodeError:
        print(f"     ⚠️  LLM returned invalid JSON: {raw[:80]}")
        return None, raw
    except Exception as e:
        print(f"     ⚠️  API error: {e}")
        time.sleep(2)
        return None, ""

# ─── Rule-Based Fallback ──────────────────────────────────────────────────────

UNCERTAIN     = ["home_office", "education", "health_expense"]
CONFIDENT_YES = ["charitable_donation", "mortgage_interest", "business_travel"]
CONFIDENT_NO  = ["personal_food", "luxury_vacation", "personal_clothing", "gym_membership"]

def rule_based_episode(task_id: int, env: TaxAdvisorEnv) -> None:
    """
    Run a complete rule-based episode directly — no step tracking needed.
    This is cleaner than guessing the next step from step index.
    """
    import random
    rng = random.Random(42)

    # Step 1: always get income + receipts
    obs, _, _, _ = env.step(TaxAction(tool_name="get_income_data", arguments={}))
    income  = obs.data["wages"]
    name    = obs.data["name"]
    status  = obs.data["filing_status"]
    icon = "✓" if obs.success else "✗"
    print(f"  Step  1 {icon}  get_income_data      → {obs.message[:55]} (r=+0.05)")

    obs, _, _, _ = env.step(TaxAction(tool_name="get_receipts", arguments={}))
    receipts = obs.data.get("receipts", [])
    print(f"  Step  2 ✓  get_receipts         → {obs.message[:55]} (r=+0.05)")

    step = 3
    total_deductions = 0

    # Step 3+: classify expenses (task 1 & 2)
    if task_id in [1, 2]:
        for r in receipts:
            cat = r["category"]
            if cat in CONFIDENT_YES:
                deductible = True
            elif cat in CONFIDENT_NO:
                deductible = False
            else:
                deductible = rng.random() < 0.70  # uncertain categories

            obs, reward, done, _ = env.step(TaxAction(
                tool_name="classify_expense",
                arguments={"category": cat, "is_deductible": deductible}
            ))
            icon = "✓" if obs.success else "✗"
            print(f"  Step {step:2d} {icon}  classify_expense     → {obs.message[:55]} (r={reward:+.2f})")
            step += 1
            if deductible:
                total_deductions += r["amount"]
            if done:
                return

    # Compute tax (task 0 & 2)
    if task_id in [0, 2]:
        obs, reward, done, _ = env.step(TaxAction(
            tool_name="compute_taxes",
            arguments={"income": income, "deductions": total_deductions}
        ))
        tax = obs.data.get("computed_tax", 0)
        icon = "✓" if obs.success else "✗"
        print(f"  Step {step:2d} {icon}  compute_taxes        → {obs.message[:55]} (r={reward:+.2f})")
        step += 1
        if done:
            return

    # Fill form + submit (task 2 only)
    if task_id == 2:
        fields = [
            ("taxpayer_name",    name),
            ("filing_status",    status),
            ("total_income",     income),
            ("total_deductions", total_deductions),
            ("tax_owed",         tax),
        ]
        for field, value in fields:
            obs, reward, done, _ = env.step(TaxAction(
                tool_name="fill_form_field",
                arguments={"field": field, "value": value}
            ))
            icon = "✓" if obs.success else "✗"
            print(f"  Step {step:2d} {icon}  fill_form_field      → {obs.message[:55]} (r={reward:+.2f})")
            step += 1

        obs, reward, done, _ = env.step(TaxAction(tool_name="submit_form", arguments={}))
        icon = "✓" if obs.success else "✗"
        print(f"  Step {step:2d} {icon}  submit_form          → {obs.message[:55]} (r={reward:+.2f})")

# ─── Episode Runner ───────────────────────────────────────────────────────────

def run_episode(task_id: int) -> float:
    """Run one full episode. Returns final score."""
    env = TaxAdvisorEnv(task_id=task_id)
    obs = env.reset()

    labels = {0: "Easy — Tax Calculation", 1: "Medium — Deduction Finder", 2: "Hard — Full Filing"}
    print(f"\n  Task: {labels[task_id]}")
    print(f"  {obs.message}")

    if client:
        # ── LLM mode ──────────────────────────────────────────────────────────
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"START: {obs.message}\nAvailable tools: {obs.data['available_tools']}"},
        ]
        total_reward = 0.0
        for step in range(MAX_STEPS):
            parsed, raw = llm_agent_step(messages)
            if not parsed:
                continue

            try:
                action = TaxAction(
                    tool_name=parsed["tool_name"],
                    arguments=parsed.get("arguments", {}),
                )
            except Exception as e:
                print(f"     ⚠️  Bad action: {e}")
                continue

            obs, reward, done, _ = env.step(action)
            total_reward += reward
            icon = "✓" if obs.success else "✗"
            print(f"  Step {step+1:2d} {icon}  {action.tool_name:20s} → {obs.message[:55]} (r={reward:+.2f})")

            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    f"Observation: {obs.message}\n"
                    f"Data: {json.dumps(obs.data)}\n"
                    f"Reward: {reward:+.2f} | Done: {done}\n"
                    f"State: fields_filled={env.state().fields_filled}, "
                    f"deductions_found={env.state().deductions_found}, "
                    f"submitted={env.state().submitted}\n"
                    f"What is your next action?"
                ),
            })
            if done:
                print(f"  ✅ Done at step {step+1}")
                break
    else:
        # ── Rule-based fallback mode ───────────────────────────────────────────
        rule_based_episode(task_id, env)

    score = grade_task(env)
    print(f"  📊 Score: {score:.4f}")
    return score

# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    mode = "LLM agent" if client else "rule-based fallback"
    print(f"\n{'='*55}")
    print(f"  TaxAdvisorEnv — Inference ({mode})")
    print(f"{'='*55}")
    if client:
        print(f"  Model : {MODEL_NAME}")
        print(f"  API   : {API_BASE_URL}")

    scores = []
    for task_id in [0, 1, 2]:
        print(f"\n{'─'*55}")
        print(f"  TASK {task_id}")
        print(f"{'─'*55}")
        score = run_episode(task_id)
        scores.append(score)

    print(f"\n{'='*55}")
    print("  FINAL RESULTS")
    print(f"{'='*55}")
    labels = ["Easy  ", "Medium", "Hard  "]
    for i, (label, score) in enumerate(zip(labels, scores)):
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  Task {i} {label} [{bar}] {score:.4f}")
    overall = sum(scores) / len(scores)
    print(f"\n  Overall: {overall:.4f}")

    if not client:
        print(f"\n  To run with a real LLM:")
        print(f"  1. Get free token: https://huggingface.co/settings/tokens")
        print(f"  2. export HF_TOKEN=your_token")
        print(f"  3. python inference.py")

if __name__ == "__main__":
    run()
