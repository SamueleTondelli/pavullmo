import json
import os
from pathlib import Path

models = ["35m_balanced_300m", "35m_knowledge_300m", "35m_web_300m"]
ds = ["bal", "knw", "web"]
score_path = Path("benchmarks/scores")
categories = [
    "icl",
    "knowledge",
    "language-manipulation",
    "math",
    "math-languages",
    "reasoning",
]
for c in categories:
    benchmarks = [Path(p).name for p in os.listdir(score_path / c / models[0])]
    overall = {"bal": 0, "knw": 0, "web": 0}
    print(f"Category: {c}")
    for b in benchmarks:
        ranking = []
        for i, m in enumerate(models):
            with open(score_path / c / m / b) as f:
                score = json.load(f)
            ranking.append((ds[i], score["metrics"]["macro_bidirectional_accuracy"]))
            overall[ds[i]] += score["metrics"]["macro_bidirectional_accuracy"]
        ranking = sorted(ranking, key=lambda x: x[1], reverse=True)
        print(f"{b + ' ' * (50 - len(b))}: ", end="")
        for r in ranking:
            print(f"{r[0]}: {r[1]:.5f}, ", end="")
        print("")

    overall_ranking = [(d, overall[d] / len(benchmarks)) for d in overall]
    overall_ranking = sorted(overall_ranking, key=lambda x: x[1], reverse=True)
    print(f"overall{' ' * (50 - len('overall'))}: ", end="")
    for r in overall_ranking:
        print(f"{r[0]}: {r[1]:.5f}, ", end="")
    print("\n")
