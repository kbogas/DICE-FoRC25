""" 
These are mainly helper functions for the FoRC dataset.
"""

import json
import os
from ast import literal_eval

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def parse_taxonomy_json(path_to_taxonomy_json: str, level=1) -> pd.DataFrame:
    """
    Helper function to parse the taxonomy json
    Args:
        path_to_taxonomy_json (str): Path to taxonomy json.
        level (int, optional): Starting level point. Defaults to 1.

    Returns:
        pd.DataFrame: The dataframe with the taxonomy in a dataframe
        with the following columns ["index", "level", "name", "direct_children", "direct_children_indices"]
    """
    with open(path_to_taxonomy_json, "r") as f:
        data = json.load(f)

    result: list[tuple] = []
    queue = [(data, level)]

    while queue:
        node, lvl = queue.pop(0)
        if lvl > 1:  # Ignore root level ("NLP")
            children = (
                [child["name"] for child in node["children"]]
                if "children" in node
                else []
            )
            result.append((len(result), lvl, node["name"].strip(), children))

        if "children" in node:
            for child in node["children"]:
                queue.append((child, lvl + 1))

    tax_df = pd.DataFrame(result, columns=["index", "level", "name", "direct_children"])
    tax_df["level"] = tax_df["level"] - 1
    label2index = dict(zip(tax_df.name, tax_df.index))
    tax_df["direct_children_indices"] = tax_df["direct_children"].apply(
        lambda x: [label2index[item] for item in x]
    )

    return tax_df


def get_labels_per_sample(df: pd.DataFrame, tax_df: pd.DataFrame) -> np.ndarray:
    """Helper function to generate a many-hot-encoded Num_Samples X Num_Labels matrix
    for a given dataframe and taxonomy df.


    Args:
        df (pd.DataFrame): The input dataframe.
        tax_df (pd.DataFrame): The taxonomy dataframe.

    Returns:
        np.ndarray: Many-hot encoded array of same Num_Samples X Num_Labels
    """
    labels = []
    for level in range(1, 4):
        cur_level_labels = df[f"Level{str(level)}"]
        ohe = CountVectorizer(
            preprocessor=lambda x: x,
            tokenizer=lambda x: x,
            vocabulary=tax_df[tax_df["level"] == level]["name"],
            token_pattern=None,
            binary=True,
        )
        labels.append(
            ohe.transform(
                [
                    literal_eval(list_) if list_ else list_
                    for list_ in cur_level_labels.tolist()
                ]
            ).toarray()
        )
    return np.hstack((labels))


def load_data(path_to_forc_data_folder: str) -> dict[str, pd.DataFrame]:
    """Simple loading function

    Args:
        path_to_forc_data_folder (str): Path to folder containing the train,val and test.csv
    Returns:
        dict[str, pd.DataFrame]:
    """
    all_dfs = {}
    for name in ["train", "val", "test"]:
        name_to_load = (
            os.path.join(
                path_to_forc_data_folder,
                name,
            )
            + ".csv"
        )
        all_dfs[name] = pd.read_csv(name_to_load)

    # add [] lists in nulls for literal_eval
    for level in range(1, 4):
        for name, df in all_dfs.items():
            isnull = df[f"Level{level}"].isnull()
            df[f"Level{level}"] = df[f"Level{level}"].apply(
                lambda x: [] if pd.isnull(x) else x
            )
            all_dfs[name] = df

    return all_dfs


from sklearn.metrics import f1_score, precision_score, recall_score


def get_scores(y_test, y_pred):
    """Helper function to generate scores. Used with sklearn syntax"""
    scores = {}
    for score_name in ["macro", "micro", "weighted"]:
        f1 = f1_score(y_test, y_pred, average=score_name)
        # print(f"F1-{score_name}: {f1:.4f}")
        scores[score_name] = f1

    prec = precision_score(y_test, y_pred, average="macro")
    # print(f"Precision (macro): {prec:.4f}")
    scores["prec_macro"] = prec

    rec = recall_score(y_test, y_pred, average="macro")
    # print(f"Recall (macro): {rec:.4f}")
    scores["rec_macro"] = rec

    prec = precision_score(y_test, y_pred, average="micro")
    # print(f"Precision (micro): {prec:.4f}")
    scores["prec"] = prec

    rec = recall_score(y_test, y_pred, average="micro")
    # print(f"Recall (micro): {rec:.4f}")
    scores["rec"] = rec
    return scores
