from env import TaxAdvisorEnv, TaxAction, grade_task
import os

# ---- OpenAI Setup ----
client = None
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
try:
    from openai import OpenAI
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    if API_KEY:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        print("✅ OpenAI client initialized", flush=True)
    else:
        print("⚠️ No API key found, running rule-based mode", flush=True)
except Exception as e:
    print("⚠️ OpenAI setup failed:", e, flush=True)
    client = None

TASK_NAMES = {0: "tax_calculation", 1: "deduction_finder", 2: "complete_filing"}

def run():
    print("\n🚀 Starting Tax Filing Agent for ALL TASKS\n", flush=True)

    for task_id in [0, 1, 2]:
        task_name = TASK_NAMES[task_id]
        print(f"[START] task={task_name}", flush=True)

        env = TaxAdvisorEnv(task_id=task_id)
        obs = env.reset()
        step_num = 0
        total_reward = 0.0

        obs, reward, _, _ = env.step(TaxAction(tool_name="get_income_data", arguments={}))
        income = obs.data["wages"]
        name   = obs.data["name"]
        status = obs.data["filing_status"]
        step_num += 1; total_reward += reward
        print(f"[STEP] step={step_num} action=get_income_data reward={reward}", flush=True)

        obs, reward, _, _ = env.step(TaxAction(tool_name="get_receipts", arguments={}))
        receipts = obs.data.get("receipts", [])
        step_num += 1; total_reward += reward
        print(f"[STEP] step={step_num} action=get_receipts reward={reward}", flush=True)

        total_deductions = 0
        if task_id in [1, 2]:
            deductible_categories = [
                "home_office", "education", "health_expense",
                "charitable_donation", "mortgage_interest", "business_travel"
            ]
            for r in receipts:
                category = r["category"]
                is_deductible = category in deductible_categories
                obs, reward, _, _ = env.step(TaxAction(
                    tool_name="classify_expense",
                    arguments={"category": category, "is_deductible": is_deductible}
                ))
                step_num += 1; total_reward += reward
                print(f"[STEP] step={step_num} action=classify_expense category={category} reward={reward}", flush=True)
                if is_deductible:
                    total_deductions += r["amount"]

        tax = 0
        if task_id in [0, 2]:
            obs, reward, _, _ = env.step(TaxAction(
                tool_name="compute_taxes",
                arguments={"income": income, "deductions": total_deductions}
            ))
            tax = obs.data.get("computed_tax", 0)
            step_num += 1; total_reward += reward
            print(f"[STEP] step={step_num} action=compute_taxes reward={reward}", flush=True)

        if task_id == 2:
            for field, value in [
                ("taxpayer_name", name), ("filing_status", status),
                ("total_income", income), ("total_deductions", total_deductions),
                ("tax_owed", tax),
            ]:
                obs, reward, _, _ = env.step(TaxAction(
                    tool_name="fill_form_field",
                    arguments={"field": field, "value": value}
                ))
                step_num += 1; total_reward += reward
                print(f"[STEP] step={step_num} action=fill_form_field field={field} reward={reward}", flush=True)

            obs, reward, _, _ = env.step(TaxAction(tool_name="submit_form", arguments={}))
            step_num += 1; total_reward += reward
            print(f"[STEP] step={step_num} action=submit_form reward={reward}", flush=True)

        score = grade_task(env)
        print(f"[END] task={task_name} score={score} steps={step_num}", flush=True)
        print(f"✅ Task {task_id} Score: {score}\n", flush=True)

if __name__ == "__main__":
    run()