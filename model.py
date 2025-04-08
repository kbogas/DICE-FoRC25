import numpy as np
import pandas as pd
import torch
import tqdm
from fuzzywuzzy import process
from ollama import ChatResponse, chat
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator


class FlatLabelList(BaseModel):
    labels: list[str]


PROMPT_TO_USE = """

You are a NLP reasearcher writing a NLP-focused literature survey paper. 
You need to annotate papers with labels that are related to the topic in the manuscript.
Each paper can have multiple labels usually around 3 and at least 1.
Output only comma-separated list of labels for this paper.

These are the details of the paper:

{sample_text} .

Select among these labels from the most similar papers to the manuscript with their relevance: {similar_labels_freq}. 
"""


class LLM_NN(BaseEstimator):

    def __init__(
        self,
        label_list: list[str],
        sentence_embedder_name: str = "all-MiniLM-L6-v2",
        top_k: int = 19,
        threshold: float = 0.01,
        llm_name: str = "gemma3:12b",
        prompt: str = PROMPT_TO_USE,
        answer_template: BaseModel = FlatLabelList,
        index2father: dict = {},
    ):
        self.sentence_embedder_name = sentence_embedder_name
        self.top_k = top_k
        self.threshold = threshold
        self.sentence_encoder = SentenceTransformer(sentence_embedder_name)
        self.llm_name = llm_name
        self.label_list = label_list
        self.prompt = prompt
        self.answer_template = answer_template
        self.index2father = index2father

    def fit(self, X, y):
        self.X_train = self.sentence_encoder.encode(X)
        self.y_train = y
        return self

    def predict_proba_knn(self, X) -> np.ndarray:  # -> Any:
        X_test = self.sentence_encoder.encode(X)
        similarity = self.sentence_encoder.similarity(X_test, self.X_train)
        top_k_indices = torch.argsort(similarity, axis=1, descending=True)[
            :, : self.top_k
        ].numpy()

        # Weighted neighbor aggregation
        neighbor_labels = self.y_train[top_k_indices]
        similarity_scores = np.take_along_axis(similarity, top_k_indices, axis=1)

        norm_sim = similarity_scores / similarity_scores.sum(-1).reshape(-1, 1)
        probas = np.einsum("sn, snl->snl", norm_sim.numpy(), neighbor_labels).sum(
            axis=1
        )
        return probas

    def predict_knn(self, X) -> np.ndarray:
        probas = self.predict_proba_knn(X)
        return (probas > self.threshold).astype(int)

    def predict_with_labels(self, X, y_pred=[]) -> list[list[str]]:
        if len(y_pred) == 0:
            y_pred = self.predict(X)
        test_labels = []
        for pred_np in y_pred:
            cur_label_indices = [
                self.label_list[index] for index in pred_np.flatten().nonzero()[0]
            ]
            test_labels.append(cur_label_indices)
        return test_labels

    def predict(self, X: np.ndarray) -> np.ndarray:

        df_test = pd.DataFrame(X, columns=["text"])
        test_proba = self.predict_proba_knn(X)
        preds_df = pd.DataFrame((test_proba > self.threshold).astype(int).nonzero()).T
        preds_df.columns = ["sample", "label_index"]
        preds_df["scores"] = test_proba[preds_df["sample"], preds_df["label_index"]]
        preds_df["label"] = [self.label_list[item] for item in preds_df["label_index"]]
        df_test["similar_labels"] = [
            ", ".join(list(set(items)))
            for items in preds_df.groupby("sample")["label"].agg(list)
        ]

        grouped = (
            preds_df.groupby(["sample", "label"])["scores"]
            .agg("max")
            .reset_index()
            .set_index("sample")
        )
        vals = []
        for k in range(df_test.shape[0]):
            cur = grouped.loc[k].reset_index(drop=True)
            if isinstance(cur, pd.Series):
                cur = pd.DataFrame(cur).T
                cur.columns = ["label", "scores"]
            cur = cur.sort_values("scores", ascending=False)
            cur = cur[cur["scores"] > self.threshold]
            vals.append(str(dict(zip(cur["label"], cur["scores"]))))

        df_test["similar_labels_freq"] = vals
        preds = []
        for i, row in tqdm.tqdm(
            df_test.iterrows(), total=len(df_test), desc="Predicting on test..."
        ):
            pred_np = np.zeros(len(self.label_list))
            pred_labels = self.predict_per_sample(row)
            pred_indices = [self.label_list.index(label) for label in pred_labels]
            pred_np[np.array(pred_indices)] = 1
            preds.append(pred_np)
        return np.array(preds)

    def predict_per_sample(self, row):
        prompt_instantiated = self.prompt.format(
            sample_text=row["text"], similar_labels_freq=row["similar_labels_freq"]
        )
        response: ChatResponse = chat(
            model=self.llm_name,
            messages=[
                {"role": "user", "content": prompt_instantiated, "temperature": 0.0}
            ],
            format=self.answer_template.model_json_schema(),  # Use Pydantic to generate the schema or format=schema
        )
        out = response["message"]["content"]
        # out

        out_checked = self.answer_template.model_validate_json(out)
        all_labels = []
        found_labels = out_checked.model_dump()["labels"]
        found_labels = [
            process.extractOne(found_label, self.label_list)[0]
            for found_label in found_labels
        ]
        for true_label in found_labels:
            cur = [true_label]
            if len(self.index2father) > 0:
                try:
                    father = self.index2father[self.label_list.index(true_label)]
                    cur.append(self.label_list[father])
                    try:
                        grandfather = self.index2father[father]
                        cur.append(self.label_list[grandfather])
                    except KeyError:
                        pass
                except KeyError:
                    pass
            all_labels.extend(cur)

        return list(set(all_labels))
