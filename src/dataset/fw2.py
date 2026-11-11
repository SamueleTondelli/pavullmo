from datasets import load_dataset
import json

# fw = load_dataset(
#     "HuggingFaceFW/fineweb-2", name="ita_Latn", split="train", streaming=True
# )
# ds = iter(fw)
# docs = []
# for _ in range(1000):
#     doc = next(ds)
#     print(f"score: {doc['language_score']}, text: {doc['text'][:100]}")
#     docs.append(doc)


# with open("fw2_dump.json", "w") as f:
#     json.dump(docs, f)


with open("fw2_dump.json", "r") as f:
    docs = json.load(f)


scores = [d["language_score"] for d in docs]
print(min(scores))
print(max(scores))

for d in docs:
    if d["language_score"] < 0.9:
        print(f"score: {d['language_score']}, text: {d['text'][:500]}")
