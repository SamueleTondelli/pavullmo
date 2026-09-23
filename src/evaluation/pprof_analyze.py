import json
import numpy as np
import sys

with open(sys.argv[1]) as f:
    score = json.load(f)

pos_distribution = {0: 0, 1: 0, 2: 0, 3: 0}
non_repeated_sets = 0
full_sets = 0
target_accuracies = []
for s in score["sets"]:
    all_ans = set()
    wins = 0
    for i, sv in enumerate(s["mean_token_logprob_matrix"]):
        ans = np.argmax(sv)
        pos_distribution[ans] += 1
        all_ans.add(ans)

        for j in range(4):
            if i == j:
                continue
            if sv[i] > sv[j]:
                wins += 1

    target_accuracies.append(wins / 12)
    if len(all_ans) > 1:
        non_repeated_sets += 1
    if len(all_ans) == 4:
        full_sets += 1


total_targets = len(score["sets"]) * 4
for i in range(4):
    print(
        f"Target {i} was chosen {pos_distribution[i] / total_targets * 100}% of the times"
    )

print(
    f"{non_repeated_sets / len(score['sets']) * 100}% of sets didn't answer all with same target"
)
print(f"{full_sets / len(score['sets']) * 100}% of sets used all targets")
print(f"Overall context-based accuracy: {score['metrics']['macro_pairwise_accuracy']}")
print(
    f"Overall target-based accuracy: {sum(target_accuracies) / len(target_accuracies)}"
)
