"""
inference.py — LLM Agent for TaxAdvisorEnv
Uses the LiteLLM proxy injected by the validator via API_BASE_URL and API_KEY.
"""
import os
import json
import time
from env import TaxAdvisorEnv, TaxAction, grade_task

# ── API setup — use exactly what the validator injects ────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
MODEL_NAME   = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

from openai import OpenAI
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
print(f"✅ LLM client ready → {MODEL_NAME}", flush=True)

# ── System prompt — describes tools, does NOT give away answers ───────────────
SYSTEM_PROMPT = """You are a tax filing AI agent. Your job is to complete tax tasks
by calling one tool at a time and reasoning from the observations you receive.

Available tools:
- get_income_data       : {}
- get_receipts          : {}
- search_tax_code       : {"query": "<topic>"}
- classify_expense      : {"category": "<name>", "is_deductible": true or false}
- compute_taxes         : {"income": <number>, "deductions": <number>}
- fill_form_field       : {"field": "<name>", "value": <value>}
  Valid fields: taxpayer_name, filing_status, total_income, total_deductions, tax_owed
- submit_form           : {}

Strategy:
1. Always start with get_income_data and get_receipts to understand the situation
2. Use search_tax_code to look up whether an expense is deductible before classifying
3. For Task 2: fill ALL form fields before calling submit_form

Respond with ONLY a raw JSON object, no explanation:
{"tool_name": "<tool>", "arguments": {}}"""

TASK_NAMES = {0: "tax_calculation", 1: "deduction_finder", 2: "complete_filing"}

def llm_step(messages: list):
    """Ask LLM for next action. Returns (parsed_dict, raw_string)."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
            max_tokens=150,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw), raw
    except Exception as e:
        print(f"⚠️ LLM error: {e}", flush=True)
        time.sleep(2)
        return None, ""

def run():
    print("\n🚀 TaxAdvisorEnv — LLM Agent\n", flush=True)

    for task_id in [0, 1, 2]:
        task_name = TASK_NAMES[task_id]
        print(f"[START] task={task_name}", flush=True)

        env = TaxAdvisorEnv(task_id=task_id)
        obs = env.reset()

        # Build initial message for the LLM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": (
                f"Task: {obs.message}\n"
                f"Available tools: {obs.data['available_tools']}\n"
                f"Begin."
            )},
        ]

        step_num = 0
        for _ in range(25):
            parsed, raw = llm_step(messages)
            if not parsed:
                continue

            try:
                action = TaxAction(
                    tool_name=parsed["tool_name"],
                    arguments=parsed.get("arguments", {}),
                )
            except Exception as e:
                print(f"⚠️ Bad action format: {e}", flush=True)
                continue

            obs, reward, done, _ = env.step(action)
            step_num += 1

            # Required structured output
            print(f"[STEP] step={step_num} action={action.tool_name} reward={reward}", flush=True)

            # Feed observation back to LLM
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": (
                f"Observation: {obs.message}\n"
                f"Data: {json.dumps(obs.data)}\n"
                f"Reward: {reward} | Done: {done}\n"
                f"State: fields_filled={env.state().fields_filled}, "
                f"deductions_found={env.state().deductions_found}, "
                f"submitted={env.state().submitted}\n"
                f"What is your next action?"
            )})

            if done:
                break

        score = grade_task(env)
        print(f"[END] task={task_name} score={score} steps={step_num}", flush=True)
        print(f"✅ Task {task_id} Score: {score}\n", flush=True)

if __name__ == "__main__":
    run()