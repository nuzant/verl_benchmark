import argparse
import os
import json
from typing import Dict, List, Optional, Any

import pandas as pd
# from verl.utils.hdfs_io import copy, makedirs
# from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    # load jsonl dataset
    assert file_path.endswith(".jsonl")
    if not os.path.exists(file_path):
        raise ValueError(f"Dataset file not found: {file_path}")
        
    try:
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                data.append(json.loads(line))
        return data
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {file_path}")
    except Exception as e:
        raise ValueError(f"Error loading dataset: {str(e)}")


# def extract_solution(solution_str: str) -> str:
#     """Extract the final boxed solution from a solution string.

#     Args:
#         solution_str: Raw solution string that may contain multiple boxed answers

#     Returns:
#         The final boxed answer with box notation removed
#     """
#     return remove_boxed(last_boxed_only_string(solution_str))


def make_map_fn(split: str):
    """Create a mapping function to process dataset examples.

    Args:
        split: Dataset split name ('train' or 'test')

    Returns:
        Function that processes individual dataset examples
    """
    def process_fn(example: Dict[str, Any], idx: int, instruction: str = None) -> Optional[Dict[str, Any]]:
        question = example.pop('prompt')
        
        # if instruction is None:
        #     instruction = "Let's think step by step and output the final answer within \\boxed{}."
        if instruction is not None:
            question = f"{question} {instruction}"
        answer = example.pop('solutions')
        if isinstance(answer, list):
            answer = answer[0]

        data = {
            "data_source": "",
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'split': split,
                'index': idx
            }
        }
        return data
    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process datasets for DeepScaler training')
    parser.add_argument('--jsonl_path', default=os.path.expanduser('./data/7B_math_only.jsonl'),
                       help='Local directory to save processed datasets')
    parser.add_argument('--local_dir', default=os.path.expanduser('./data'),
                       help='Local directory to save processed datasets')
    parser.add_argument('--hdfs_dir', default=None,
                       help='Optional HDFS directory to copy datasets to')
    args = parser.parse_args()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    jsonl_path = args.jsonl_path
    
    # Make local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    # Process training data
    train_dataset = load_dataset(jsonl_path)
    train_data: List[Dict[str, Any]] = []
    process_fn = make_map_fn('train')
    for idx, example in enumerate(train_dataset):
        processed_example = process_fn(example, idx)
        if processed_example is not None:
            train_data.append(processed_example)

    # Save training dataset
    print("train data size:", len(train_data))
    train_df = pd.DataFrame(train_data)
    train_df.to_parquet(os.path.join(local_dir, 'train.parquet'))

    # Optionally copy to HDFS
    # if hdfs_dir is not None:
    #     makedirs(hdfs_dir)
    #     copy(src=local_dir, dst=hdfs_dir)