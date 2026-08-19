import os, torch, transformers, jlens
from datasets import load_dataset

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
TAG = MODEL_NAME.split("/")[-1].replace("Qwen2.5-", "").replace("-Instruct", "")

hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16).cuda()
tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = jlens.from_hf(hf, tok)

# 128-token sequences, as the paper uses. Untruncated C4 documents make the
# Jacobian estimator quadratically more expensive -- this is the difference
# between ~20 minutes and ~2 hours.
ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
prompts = []
for row in ds:
    t = tok.encode(row["text"])[:128]
    if len(t) == 128:
        prompts.append(tok.decode(t))
    if len(prompts) == 100:
        break
print(f"{len(prompts)} prompts, 128 tokens each")

lens = jlens.fit(model, prompts=prompts, checkpoint_path=f"data/ckpt_{TAG}.pt")
lens.save(f"data/jacobian_lens_{TAG}.pt")
print(f"saved data/jacobian_lens_{TAG}.pt")