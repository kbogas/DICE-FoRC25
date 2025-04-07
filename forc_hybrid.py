import numpy as np
from sklearn.feature_extraction import text

from model import LLM_NN
from utils import get_labels_per_sample, get_scores, load_data, parse_taxonomy_json

stop = text.ENGLISH_STOP_WORDS
pat = r"\b(?:{})\b".format("|".join(stop))


num_test_to_run_on = 5
data_to_use_in_train = ["abstract", "label_string", "title"]
data_to_use_in_test = ["abstract", "title"]

path_to_data = "/home/kbougatiotis/GIT/forc4cl-corpus/DICE-FoRC25/data"
path_to_taxonomy_json = (
    "/home/kbougatiotis/GIT/forc4cl-corpus/DICE-FoRC25/data/taxonomy.json"
)

# Load/process data + labels
tax_df = parse_taxonomy_json(path_to_taxonomy_json)
all_dfs = load_data(path_to_data)
all_labels_list = tax_df["name"].tolist()
label_lengths = [
    (tax_df["level"] == 1).sum(),
    (tax_df["level"] == 2).sum(),
    (tax_df["level"] == 3).sum(),
]
label_ranges = [
    (0, label_lengths[0]),
    (label_lengths[0], label_lengths[0] + label_lengths[1]),
    (label_lengths[0] + label_lengths[1], len(all_labels_list)),
]

index2children = dict(zip(tax_df.index, tax_df.direct_children_indices))
index2father = {}
for father, children in index2children.items():
    for child in children:
        index2father[child] = father
name2level = dict(zip(tax_df["name"], tax_df["level"]))


labels_train = get_labels_per_sample(all_dfs["train"], tax_df)
labels_val = get_labels_per_sample(all_dfs["val"], tax_df)


# Concatenated the label_string
all_dfs["train"]["label_string"] = [
    " ".join(np.array(all_labels_list)[y > 0]) for y in labels_train
]
all_dfs["val"]["label_string"] = [
    " ".join(np.array(all_labels_list)[y > 0]) for y in labels_val
]

# Create X_train, X_val, X_test


model = LLM_NN(label_list=all_labels_list, index2father=index2father)


X_train = (
    all_dfs["train"][data_to_use_in_train].fillna("").agg("".join, axis=1).str.lower()
)
X_train = X_train.str.replace(pat, "").str.replace(r"\s+", " ")
X_train = np.array(X_train.tolist())[:]
y_train = labels_train[:]


# X_val = all_dfs["val"][data_to_use_in_test].fillna("").agg("".join, axis=1).str.lower()
# X_val = X_val.str.replace(pat, "").str.replace(r"\s+", " ")
# X_val = np.array(X_val.tolist())

# THESE ARE RESULTS USING THE VALIDATION SET. CHANGE TO TEST IF NEEDED, BUT REMEMBER TO COMMENT OUT
# THE METRICS USED AS WE DON'T HAVE INFO ON THE LABELS.

X_test = all_dfs["val"][data_to_use_in_test].fillna("").agg("".join, axis=1).str.lower()
X_test = X_test.str.replace(pat, "").str.replace(r"\s+", " ")
X_test = np.array(X_test.tolist())[:num_test_to_run_on]
y_test = labels_val[:num_test_to_run_on]

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
scores = get_scores(y_test, y_pred)
for score, score_val in scores.items():
    print(f"{score}: {score_val:.4f}")

print("\n Indicative per sample predictions \n ")
y_pred_labels = model.predict_with_labels(X_test, y_pred)
for sample_index, labels in enumerate(y_pred_labels[:2]):
    cur_test_labels = sorted(
        [
            all_labels_list[index_nonzero]
            for index_nonzero in y_test[sample_index].flatten().nonzero()[0]
        ]
    )
    print(
        f"(Sample {sample_index + 1}): {X_test[sample_index]}\nCorrect Labels:{cur_test_labels} \nPred Labels: {sorted(labels)}"
    )
    print("\n")
