from datasets import load_dataset

ds = load_dataset("uonlp/CulturaX", "it", streaming=True, split="train")
dsiter = iter(ds)
for _ in range(10):
    d = next(dsiter)
    print(d)
