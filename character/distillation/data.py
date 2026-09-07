"""
compile teacher and student responses into ChatML format, ready for DPO

filter out broken responses or prompts that are too long
"""

import os, unicodedata
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from character.utils import constitutions
from character.constants import DATA_PATH, MODEL_PATH


def check(s):
    # check if response is not empty and ends with punctuation
    s = s.rstrip()
    return bool(s) and unicodedata.category(s[-1]).startswith("P")


def main(model: str, current_constitutions: list[str], teacher_name: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{model}")
    name = model.split("-")[0].capitalize()
    for constitution in tqdm(current_constitutions, desc=model):
        # read responses
        PATH = f"{DATA_PATH}/distillation/{constitution}.jsonl"
        if not os.path.exists(PATH):
            print(f"skipping {constitution}: no teacher responses at {PATH}")
            continue
        responses = pd.read_json(PATH, orient="records", lines=True).dropna()
        if model not in responses.columns:
            print(f"skipping {constitution}: no {model} responses in {PATH}")
            continue

        # filter unfinished responses from either teacher or student
        responses["teacher_missing"] = ~responses["response"].apply(check)
        responses["student_missing"] = ~responses[model].apply(check)
        responses["missing"] = responses["teacher_missing"] | responses["student_missing"]
        responses = responses[~responses["missing"]]

        # ChatML format, chosen/rejected for DPO
        data = pd.DataFrame(columns=["chosen", "rejected"])
        data["chosen"] = responses.apply(
            lambda row: [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"].replace(teacher_name, name)},
            ],
            axis=1,
        )
        data["rejected"] = responses.apply(
            lambda row: [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row[model]},
            ],
            axis=1,
        )

        # filter out prompts that are too long
        data["c_prompt"] = data["chosen"].apply(
            lambda x: tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)
        )
        data["r_prompt"] = data["rejected"].apply(
            lambda x: tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)
        )
        data["c_length"] = data["c_prompt"].apply(lambda x: len(tokenizer.encode(x)))
        data["r_length"] = data["r_prompt"].apply(lambda x: len(tokenizer.encode(x)))
        data["max_length"] = data[["c_length", "r_length"]].max(axis=1)
        data = data[data["max_length"] <= 1024]
        data = data[["chosen", "rejected"]]

        # save
        outpath = f"{DATA_PATH}/dpo/{model}/{constitution}.jsonl"
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        data.to_json(outpath, orient="records", lines=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--constitution", type=str, required=False, default=None)
    # nothing in this stage samples, so --seed is accepted only so the runbook can pass
    # the same seed to every data stage
    parser.add_argument("--seed", type=int, required=False, default=0)
    parser.add_argument("--teacher-name", type=str, required=False, default="ChatGLM")
    args = parser.parse_args()
    main(
        args.model,
        [args.constitution] if args.constitution else constitutions,
        args.teacher_name,
    )
