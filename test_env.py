from env import TaxAdvisorEnv, TaxAction, grade_task

# Choose task (0=easy, 1=medium, 2=hard)
env = TaxAdvisorEnv(task_id=0)

obs = env.reset()
print("START:", obs.message)

# Step 1: Get income
action = TaxAction(tool_name="get_income_data", arguments={})
obs, reward, done, info = env.step(action)
print(obs.message, obs.data, reward)

# Step 2: Compute tax
action = TaxAction(
    tool_name="compute_taxes",
    arguments={"income": 75000, "deductions": 0}
)
obs, reward, done, info = env.step(action)
print(obs.message, obs.data, reward)

# Final Score
score = grade_task(env)
print("FINAL SCORE:", score)