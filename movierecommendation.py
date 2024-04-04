import numpy as np
import random

items = ["Movie A", "Movie B", "Movie C", "Movie D", "Movie E"]
user_interactions = {
    "User 1": [1, 0, 1, 0, 0],
    "User 2": [0, 1, 0, 1, 0],
    "User 3": [1, 1, 0, 0, 1],
}

learning_rate = 0.1
discount_factor = 0.9
exploration_prob = 0.2
num_episodes = 1000
q_values = {item: 0 for item in items}

for _ in range(num_episodes):
    user = random.choice(list(user_interactions.keys()))
    state = random.choice(items)
    total_reward = 0
    
    for _ in range(10):
        if np.random.rand() < exploration_prob:
            action = random.choice(items)
        else:
            action = max(q_values, key=lambda x: q_values[x])
        reward = user_interactions[user][items.index(action)]
        q_values[state] += learning_rate * (reward + discount_factor * max(q_values.values()) - q_values[state])
        state = action
        total_reward += reward
        
    print(f"Episode {_ + 1}, User: {user}, Total Reward: {total_reward}")

target_user = "User 1"
best_item = max(q_values, key=lambda x: q_values[x])
print(f"Recommended item for {target_user}: {best_item}")
