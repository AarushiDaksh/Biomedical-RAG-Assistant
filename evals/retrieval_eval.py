import pandas as pd
from rag.retriever import retrieve_chunks


def reciprocal_rank(chunks, target_doc):
    for i, c in enumerate(chunks, start=1):
        if c.document_id == target_doc:
            return 1 / i
    return 0


def run(path="evals/eval_set.csv"):
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        if r["ground_truth_document_id"] in ["NA", "TODO"]:
            continue
        chunks = retrieve_chunks(r["question"], k=5)
        docs = [c.document_id for c in chunks]
        rows.append({
            "question": r["question"],
            "hit@1": int(r["ground_truth_document_id"] in docs[:1]),
            "hit@3": int(r["ground_truth_document_id"] in docs[:3]),
            "hit@5": int(r["ground_truth_document_id"] in docs[:5]),
            "rr": reciprocal_rank(chunks, r["ground_truth_document_id"]),
        })
    out = pd.DataFrame(rows)
    print(out)
    if len(out):
        print("Hit@1", out["hit@1"].mean(), "Hit@3", out["hit@3"].mean(), "Hit@5", out["hit@5"].mean(), "MRR", out["rr"].mean())

if __name__ == "__main__":
    run()
