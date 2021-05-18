#!/usr/bin/env python
"""Homograph disambiguation training and evaluation."""

import csv
import glob

from typing import Dict, List, Tuple

import nltk 
import sklearn.feature_extraction 
import sklearn.linear_model


FeatureVector = Dict[str, str]
FeatureVectors = List[FeatureVector]


TRAIN_TSV = "data/train/*.tsv"


def extract_features(
    sentence: str, homograph: str, start: int, end: int
) -> FeatureVector:
    """Extracts  feature vector for a single sentence."""
    # There is some tricky stuff to find the target homograph word here.
    sentence_b = sentence.encode("utf8")
    left = sentence_b[:start]
    target = b"^" + sentence_b[start:end] + b"^"
    right = sentence_b[end:]
    sentence = (left + target + right).decode("utf8")
    tokens = nltk.word_tokenize(sentence)
    t = -1
    for (i, token) in enumerate(tokens):
        if token.count("^") == 2:
            t = i
            break
    assert t != -1, f"target homograph {homograph!r} not found"
    target = tokens[t].replace("_", "")
    # Now onto feature extraction.
    features: Dict[str, str] = {}
    # TODO: add features to the feature dictionary here using `token`, its
    # index `t`, and the list of tokens `tokens`.
    
    if tokens[t] != tokens[1]:
        features["t-1"] = tokens[t-1]
    elif tokens[t] != tokens[-1]:
        features["t+1"] = tokens[t+1]
    elif tokens[t] != tokens[1] or tokens[2]:
        features["t-2"] = tokens[t-2]
    elif tokens[t] != tokens[-1] or tokens[-2]:
        features["t+2"] = tokens[t+2]
    elif tokens[t] != tokens[1] or tokens[2]:
        features["t-2^t-1"] = tokens[t-2^t-1]
    elif tokens[t] != tokens[-1] or tokens[-2]:
        features["t+1^t+2"] = tokens[t+1^t+2]
    elif tokens[t] != tokens[1] or tokens[-1]:
        features["t-1^t+1"] = tokens[t-1^t+1]
    elif tokens[t] == tokens.casefold[t]:
        features["cap(t)"] = "lower"
    return features


def extract_features_file(path: str) -> Tuple[FeatureVectors, List[str]]:
    """Extracts feature vectors for an entire TSV file."""
    features: FeatureVectors = []
    labels: List[str] = []
    with open(path, "r") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            labels.append(row["wordid"])
            features.append(
                extract_features(
                    row["sentence"],
                    row["homograph"],
                    int(row["start"]),
                    int(row["end"]),
                )
            )
    return features, labels


def main() -> None:
    correct: List[int] = []
    size: List[int] = []
    for train_path in glob.iglob(TRAIN_TSV):
        # TODO: Extract training features and labels using
        # `extract_features_files`.
        features, labels = extract_features_file(train_path)
        # TODO: Create a DictVectorizer object.
        vectorizer = sklearn.feature_extraction.DictVectorizer()
        # TODO: One-hot-encode the features using the object's `fit_transform`.
        train_feature_vect = vectorizer.fit_transform(features)
        # TODO: Create a LogisticRegression object.
        model = sklearn.linear_model.LogisticRegression(
            solver="liblinear"
        )
        # TODO: Fit the model using the object's `fit`, the vectorized features,
        # and the labels.
        model.fit(train_feature_vect, labels)
        # TODO: compute the path of the test file for the current homograph.
    TEST_TSV = "data/test/*.tsv"
    for test_path in glob.iglob(TEST_TSV):
        # TODO: Extract test features and labels using 
        # `extract_features_file`.
        test_feat, test_labels = extract_features_file(test_path)
        # TODO: One-hot-encode the features using the DictVectorizer's `transform`.
        test_feat_vect = vectorizer.transform(test_feat)
        # TODO: Compute the number of correct predictions and append it to
        # `correct`.
        predicted = model.predict(test_feat_vect)
        correct.append(predicted)
        # TODO: Append the size of the test set to `size`.
        size.append(test_labels)
    # Accuracies.
    # TODO: print micro-averaged accuracy.
    print(len(correct)/len(size))

    # TODO: print macro-averaged accuracy.
    ...


if __name__ == "__main__":
    main()
