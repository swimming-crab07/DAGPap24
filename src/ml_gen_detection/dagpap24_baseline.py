import argparse
import gc
import json
import logging
import os
import random
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.utils import get_project_root


from src.ml_gen_detection.hf_token_classification import main as hf_token_classification

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

"""
This script performs span prediction and classification with bert models.
"""


def chunk_tokens_labels(df: pd.DataFrame, max_length: int, model_name_or_path: str, cache_dir: str,):
    """
    This function chunks tokens and their respective labels to
    max_length token length
    """
    index_list = []
    tokens_list = []
    labels_list = []
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        use_fast=True,
    )
    for index, row in tqdm(df.iterrows(), total=len(df)):
        tokenized_inputs = tokenizer(
            row["tokens"],
            # We use this argument because the texts in our dataset are lists
            # of words (with a label for each word).
            is_split_into_words=True,
        )
        idx = -1
        last_truc_word = -1
        word_ids = tokenized_inputs.word_ids()[1:-1]
        if len(tokenized_inputs["input_ids"]) > max_length:
            remaining_tokens = row["tokens"]
            remaining_labels = row["token_label_ids"]
            idx = idx + max_length - 2
            truc_word = word_ids[idx] - last_truc_word -1
            last_truc_word = word_ids[idx] - 1 
            while word_ids[idx] == word_ids[idx-1]:
                idx = idx - 1
            # While the remaining list is larger than the number of word we need to trucate to fit max_length tokens,
            # truncate and append
            while len(remaining_labels) > truc_word:
                if truc_word == 0:
                    while word_ids[idx] == word_ids[idx+1]:
                        idx = idx + 1
                        if idx > len(word_ids) - 2:
                            break
                    idx = idx + 1
                    if idx > len(word_ids) - 1:
                        break
                    last_truc_word = word_ids[idx] - 1
                    truc_word = 1
                index_list.append(index)
                tokens_list.append(remaining_tokens[:truc_word])
                labels_list.append(remaining_labels[:truc_word])
                remaining_tokens = remaining_tokens[truc_word:]
                remaining_labels = remaining_labels[truc_word:]
                idx = idx + max_length - 3
                if idx < len(word_ids):
                    truc_word = word_ids[idx] - last_truc_word -1
                    last_truc_word = word_ids[idx] - 1
                    while word_ids[idx] == word_ids[idx-1]:
                        idx = idx - 1
                # Average the length of the last chip with the length of the penultimate chip
                else:
                    remaining_tokens = list(tokens_list[-1]) + list(remaining_tokens)
                    remaining_labels = list(labels_list[-1]) + list(remaining_labels)
                    last_truc_word = last_truc_word - len(tokens_list[-1])
                    idx = idx + 3 - max_length
                    idx = idx - len(tokenizer(tokens_list[-1],is_split_into_words=True,)["input_ids"]) + 2
                    idx_m = (idx + len(word_ids) - 1) // 2
                    idx_f = idx_m
                    idx_b = idx_m
                    while word_ids[idx_f] == word_ids[idx_f-1]:
                        idx_f = idx_f - 1
                    if (len(word_ids) - 1 - idx_f) > max_length:
                        while word_ids[idx_b] == word_ids[idx_b+1]:
                            idx_b = idx_b + 1
                        idx_b = idx_b + 1
                        truc_word = word_ids[idx_b] - last_truc_word -1
                        tokens_list[-1] = remaining_tokens[:truc_word]
                        labels_list[-1] = remaining_labels[:truc_word]
                        index_list.append(index)
                        remaining_tokens = remaining_tokens[truc_word:]
                        tokens_list.append(remaining_tokens)
                        remaining_labels = remaining_labels[truc_word:]
                        labels_list.append(remaining_labels)
                        break
                    else:
                        truc_word = word_ids[idx_f] - last_truc_word -1
                        tokens_list[-1] = remaining_tokens[:truc_word]
                        labels_list[-1] = remaining_labels[:truc_word]
                        index_list.append(index)
                        remaining_tokens = remaining_tokens[truc_word:]
                        tokens_list.append(remaining_tokens)
                        remaining_labels = remaining_labels[truc_word:]
                        labels_list.append(remaining_labels)
                        break
        else:
            index_list.append(index)
            tokens_list.append(row["tokens"])
            labels_list.append(row["token_label_ids"])

    return pd.DataFrame(
        {"index": index_list, "tokens": tokens_list, "labels": labels_list}
    )


def write_df_to_json(df: pd.DataFrame, path_to_json: str):
    """
    This function writes pandas dataframes into a compatible json format
    to be used by hf_token_classification.py
    """
    index_list = df["index"].values.tolist()
    tokens_list = df["tokens"].values.tolist()
    labels_list = df["labels"].values.tolist()
    data_list = []
    for i in tqdm(range(len(tokens_list)), total=len(tokens_list)):
        index = index_list[i]
        tokens = tokens_list[i]
        labels = [str(el) for el in labels_list[i]]
        data_list.append(
            {"index": index, "tokens": tokens, "ner_tags": labels}
        )
    with open(path_to_json, "w") as f:
        f.write(json.dumps(data_list))


def prep_data(path_to_file: str, max_length: int, model_name_or_path: str, cache_dir: str, test: bool = False,):
    if test == False:
        dataset = "train"
    else:
        dataset = "test"

    logger.info(f"Loading {dataset} dataset from file")
    df = pd.read_parquet(path_to_file, engine="fastparquet")
    if df.index.name != "index":
        df.set_index("index", inplace=True)

    # the external NER Classification script needs a target column
    # for the test set as well, which is not available.
    # Therefore, we're subsidizing this column with a fake label column
    # Which we're not using anyway, since we're only using the test set
    # for predictions.
    if "token_label_ids" not in df.columns:
        df["token_label_ids"] = df["tokens"].apply(
            lambda x: np.zeros(len(x), dtype=int)
        )
    df = df[["tokens", "token_label_ids"]]

    logger.info(f"Initial {dataset} data length: {len(df)}")
    df = chunk_tokens_labels(df, max_length=max_length, model_name_or_path=model_name_or_path, cache_dir=cache_dir)
    logger.info(
        f"{dataset} data length after chunking to max {max_length} tokens: {len(df)}"
    )

    return df


def convert_parquet_data_to_json(
    input_folder_path: str,
    input_train_file_name: str,
    input_test_file_name: str,
    max_length: int,
    val_size: float = 0.1,
    output_train_file_name: str = "",
    output_val_file_name: str = "",
    output_test_file_name: str = "",
    seed: int = 0,
    model_name_or_path: str ="",
    cache_dir: str ="",
):
    """
    This function takes a parquet file with (at least) the text split and the token
    label ids, and converts it to train, validation, and test data.
    Each chunk is saved as a separate json file

    param input_folder_path: path to data dir
    param input_train_file_name: the input train file name
    param input_test_file_name: the input test file name
    param max_length: max token length
    param val_size: validation size as a fraction of the total
    param output_train_file_name: json train file name
    param output_val_file_name: json validation file name
    param output_test_file_name: json test file name

    returns: None
    """

    logger.info("Loading train and test datasets")

    # Loading and prepping train dataset
    train_df = prep_data(
        path_to_file=Path(input_folder_path) / Path(input_train_file_name),
        max_length=max_length,
        test=False,
        model_name_or_path=model_name_or_path,
        cache_dir=cache_dir,
    )
    # Loading and prepping test dataset
    test_df = prep_data(
        path_to_file=Path(input_folder_path) / Path(input_test_file_name),
        max_length=max_length,
        test=True,
        model_name_or_path=model_name_or_path,
        cache_dir=cache_dir,
    )

    logger.info("Splitting train data into train and validation splits (K-fold validation)")
    # Make the kfold object
    #folds = StratifiedKFold(n_splits=10)
    
    # Now make our splits based off of the labels. 
    # We can use `np.zeros()` here since it only works off of indices, we really care about the labels
    #splits = folds.split(np.zeros(datasets["train"].num_rows), datasets["train"]["label"])
    
    train_df, val_df = train_test_split(
        train_df, test_size=val_size, random_state=seed, shuffle=True
    )

    logger.info(f"Final train size: {len(train_df)}")
    logger.info(f"Final validation size: {len(val_df)}")
    logger.info(f"Final test size: {len(test_df)}")

    logger.info("Writing train df to json...")
    write_df_to_json(
        train_df,
        f"{input_folder_path}/{output_train_file_name}",
    )
    logger.info("Writing val df to json...")
    write_df_to_json(val_df, f"{input_folder_path}/{output_val_file_name}")
    logger.info("Writing test df to json...")
    write_df_to_json(
        test_df,
        f"{input_folder_path}/{output_test_file_name}",
    )


def convert_preds_to_original_format(
    path_to_test_data: str = "",
    path_to_test_preds: str = "",
    path_to_final_output: str = "",
):
    """
    This function takes the chunked preds and groups them into the original format
    """

    orig_test_data = pd.read_parquet(path_to_test_data, engine="fastparquet")
    if orig_test_data.index.name != "index":
        orig_test_data.set_index("index", inplace=True)

    with open(path_to_test_preds, "r") as f:
        test_preds = json.load(f)
            

    test_preds_df = pd.DataFrame(test_preds).groupby(by="index").agg(list)

    test_preds_df["preds"] = test_preds_df["predictions"].apply(
        lambda x: sum(x, [])
    )
        
    for index, row in test_preds_df.iterrows():
        assert len(row["preds"]) == len(orig_test_data.loc[index, "tokens"])

    pd.DataFrame(test_preds_df["preds"]).to_parquet(path_to_final_output)
    print(f"final dataset saved to {path_to_final_output}")

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Competition data prep")
    parser.add_argument(
        "--config_file",
        type=str,
        default="config_baseline.yml",
    )
    args = parser.parse_args()

    # loading config params
    project_root: Path = get_project_root()
    with open(str(project_root / "config" / args.config_file)) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    path_to_data_folder = str(project_root / params["data"]["path_to_data"])

    input_train_file_name = params["data"]["train_file"]
    input_test_file_name = params["data"]["test_file"]
    output_train_file_name = params["data"]["train_file_name"]
    output_val_file_name = params["data"]["validation_file_name"]
    output_test_file_name = params["data"]["test_file_name"]

    convert_parquet_data_to_json(
        input_folder_path=path_to_data_folder,
        input_train_file_name=input_train_file_name,
        input_test_file_name=input_test_file_name,
        max_length=params["bert"]["MAX_LENGTH"],
        val_size=params["environment"]["val_size"],
        output_train_file_name=output_train_file_name,
        output_val_file_name=output_val_file_name,
        output_test_file_name=output_test_file_name,
        seed=params["environment"]["SEED"],
        model_name_or_path=params["bert"]["model"],
        cache_dir=params["bert"]["cache_dir"]
    )
    # create hf_token_classification.py config file
    config_dict = {
        # "local_rank" : args.local_rank,
        "train_file": f"{path_to_data_folder}/{output_train_file_name}",
        "validation_file": f"{path_to_data_folder}/{output_val_file_name}",
        "test_file": f"{path_to_data_folder}/{output_test_file_name}",
        "output_dir": f"{path_to_data_folder}/{params['bert']['output_dir']}",
        "model_name_or_path": params["bert"]["model"],
        "cache_dir": params["bert"]["cache_dir"],
        "num_train_epochs": params["bert"]["num_train_epochs"],
        "per_device_train_batch_size": params["bert"]["per_device_train_batch_size"],
        "per_device_eval_batch_size": params["bert"]["per_device_eval_batch_size"],
        "save_steps": params["bert"]["save_steps"],
        "overwrite_output_dir": params["bert"]["overwrite_output_dir"],
        "seed": params["environment"]["SEED"],
        "do_train": params["bert"]["do_train"],
        "report_to": params["bert"]["report_to"],
        "do_eval": params["bert"]["do_eval"],
        "do_predict": params["bert"]["do_predict"],
        "preprocessing_num_workers": params["bert"]["preprocessing_num_workers"],
        "eval_accumulation_steps": params["bert"]["eval_accumulation_steps"],
        "log_level": params["bert"]["log_level"],
        "weight_decay" : 0,
        "warmup_ratio" : 0,
        "learning_rate" : 5e-5,
    }

    # save hf_token_classification.py config file
    hf_config_file_path = str(project_root / "config/config_huggingface.json")
    with open(hf_config_file_path, "w") as f:
        json.dump(config_dict, f, indent=4)
        
    hf_token_classification(json_config_file_path=hf_config_file_path)

    path_to_test_data = str(
        project_root
        / f'{params["data"]["path_to_data"]}/{input_test_file_name}'
    )
    path_to_test_preds = str(
        project_root
        / f'{params["data"]["path_to_data"]}/{params["bert"]["output_dir"]}/test_predictions.json'
    )
    path_to_final_output = str(
        project_root
        / f'{params["data"]["path_to_data"]}/preds_{params["bert"]["model"].split("/")[0]}.parquet'
    )

    convert_preds_to_original_format(
        path_to_test_data=path_to_test_data,
        path_to_test_preds=path_to_test_preds,
        path_to_final_output=path_to_final_output,
    )
