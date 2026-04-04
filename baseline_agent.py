"""
baseline_agent.py — Baseline inference script for TaxAdvisorEnv
Uses the OpenAI API client (compatible with any OpenAI-compatible endpoint).
Reads credentials from environment variable: OPENAI_API_KEY
Produces a reproducible baseline score on all 3 tasks.
"""

import os
import json
from openai import OpenAI
from env import TaxAdvisorEnv, TaxAction, grade_task

# ─── Config ──────────────────────────────────────────────────────────────────

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_STEPS = 20
NUM_PROFILES = 2   # run each task on each profile for averaging

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a tax filing AI agent. You interact with a tax environment
by calling tools one at a time. After each tool call you receive an observation.

Available tools:
- get_income_data: {}  → returns taxpayer income info
- get_receipts: {}  → returns list of expense receipts
- classify_expense: {"category": "<name>", "is_deductible": true/false}
- search_tax_code: {"query": "<topic>"}  → returns relevant tax rules
- compute_taxes: {"income": <number>, "deductions": <number>}
- fill_form_field: {"field": "<name>", "value": <value>}
  Valid fields: taxpayer_name, filing_status, total_income, total_deductions, tax_owed
- submit_form: {}  → submits the completed tax form

Always respond with a single JSON object like:
{"tool_name": "<tool>", "arguments": {}}

Do not include any explanation. Only output the JSON.
"""

# ─── Agent Loop ───────────────────────────────────────────────────────────────

def run_episode(task_id: int, profile_index: int, verbose: bool = True) -> float:
    """Run one episode and return the final score."""
    env = TaxAdvisorEnv(task_id=task_id, profile_index=profile_index)
    obs = env.reset()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{obs.message}\n\nInitial data: {json.dumps(obs.data)}"},
    ]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task {task_id} | Profile {profile_index}")
        print(f"{'='*60}")
        print(f"[ENV] {obs.message}")

    total_reward = 0.0

    for step in range(MAX_STEPS):
        # Get action from LLM
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()

        # Parse action
        try:
            parsed = json.loads(raw)
            action = TaxAction(
                tool_name=parsed["tool_name"],
                arguments=parsed.get("arguments", {}),
            )
        except Exception as e:
            if verbose:
                print(f"[Step {step+1}] Parse error: {e} | Raw: {raw}")
            break

        # Step environment
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if verbose:
            status = "✓" if obs.success else "✗"
            print(f"[Step {step+1}] {status} {action.tool_name} → {obs.message} (reward={reward:+.2f})")

        # Add to message history
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": f"Observation: {obs.message}\nData: {json.dumps(obs.data)}\nReward: {reward}\nDone: {done}",
        })

        if done:
            if verbose:
                print(f"[Done] Episode finished at step {step+1}")
            break

    score = grade_task(env)
    if verbose:
        print(f"\n[Score] Task {task_id} | Profile {profile_index} → {score:.4f}")
        print(f"[Total reward] {total_reward:.4f}")
    return score


# ─── Main: Run All 3 Tasks ────────────────────────────────────────────────────

def main():
    print("TaxAdvisorEnv — Baseline Inference Script")
    print(f"Model: {MODEL}\n")

    results = {}
    for task_id in [0, 1, 2]:
        scores = []
        for profile_index in range(NUM_PROFILES):
            score = run_episode(task_id=task_id, profile_index=profile_index, verbose=True)
            scores.append(score)
        avg = round(sum(scores) / len(scores), 4)
        results[f"task_{task_id}"] = {"scores": scores, "average": avg}
        print(f"\n>>> Task {task_id} average score: {avg:.4f}\n")

    print("\n" + "="*60)
    print("FINAL BASELINE RESULTS")
    print("="*60)
    for task, data in results.items():
        print(f"  {task}: {data['average']:.4f}  (runs: {data['scores']})")

    overall = round(
        sum(d["average"] for d in results.values()) / len(results), 4
    )
    print(f"\n  Overall average: {overall:.4f}")
    return results


if __name__ == "__main__":
    main()
