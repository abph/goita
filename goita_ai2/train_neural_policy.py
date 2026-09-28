"""Train and export the small neural candidate-ranking policy for 強化中AI2."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from goita_ai2.neural_policy import ACTION_TO_INDEX, action_tuple, record_features


DEFAULT_DATA = Path("private_data/neural_training/decisions.jsonl")
DEFAULT_MODEL = Path("goita_ai2/experimental_ai2/data/neural_policy.json")
DEFAULT_REPORT = Path("private_data/neural_training/training_report.json")


def _records(path: Path) -> list[Dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _selected_action(record: Mapping[str, Any]):
    legal = list(record["legal_actions"])
    return action_tuple(legal[int(record["selected_action_index"])])


def _arrays(records: Sequence[Mapping[str, Any]]):
    import numpy as np

    feature_names: Optional[list[str]] = None
    rows: list[list[float]] = []
    labels: list[int] = []
    for record in records:
        names, values = record_features(record)
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError("inconsistent feature schema")
        rows.append(values)
        labels.append(ACTION_TO_INDEX[_selected_action(record)])
    return feature_names or [], np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def _metrics(model, scaler, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    import numpy as np

    if not records:
        return {"records": 0}
    _names, matrix, _labels = _arrays(records)
    probabilities = model.predict_proba(scaler.transform(matrix))
    class_column = {int(class_id): index for index, class_id in enumerate(model.classes_)}
    exact = type_match = attack_match = top3 = first_baseline = 0
    for row_index, record in enumerate(records):
        legal = [action_tuple(item) for item in record["legal_actions"]]
        actual = _selected_action(record)
        ranked = sorted(
            legal,
            key=lambda action: float(
                probabilities[row_index, class_column[ACTION_TO_INDEX[action]]]
            ) if ACTION_TO_INDEX[action] in class_column else -1.0,
            reverse=True,
        )
        predicted = ranked[0]
        exact += int(predicted == actual)
        type_match += int(predicted[0] == actual[0])
        attack_match += int(predicted[0] == actual[0] and predicted[2] == actual[2])
        top3 += int(actual in ranked[:3])
        first_baseline += int(legal[0] == actual)
    count = len(records)
    return {
        "records": count,
        "exact_accuracy": round(exact / count, 6),
        "action_type_accuracy": round(type_match / count, 6),
        "attack_piece_accuracy": round(attack_match / count, 6),
        "top3_legal_accuracy": round(top3 / count, 6),
        "first_legal_baseline_accuracy": round(first_baseline / count, 6),
    }


def train_policy(
    data_path: Path,
    model_path: Path,
    report_path: Path,
    *,
    target_repeat: int = 3,
    hidden_layers: tuple[int, ...] = (96, 48),
    max_iter: int = 60,
    random_state: int = 1222,
) -> Dict[str, Any]:
    import numpy as np
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    records = _records(data_path)
    train = [item for item in records if item["split"] == "train"]
    validation = [item for item in records if item["split"] == "validation"]
    test = [item for item in records if item["split"] == "test"]
    target_train = [item for item in train if item.get("is_target_player")]
    weighted_train = train + target_train * max(0, int(target_repeat) - 1)
    feature_names, train_x, train_y = _arrays(weighted_train)
    scaler = StandardScaler()
    transformed = scaler.fit_transform(train_x)
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=7,
        random_state=random_state,
        verbose=False,
    )
    model.fit(transformed, train_y)

    evaluation = {
        "validation_all": _metrics(model, scaler, validation),
        "validation_1222": _metrics(model, scaler, [item for item in validation if item.get("is_target_player")]),
        "test_all": _metrics(model, scaler, test),
        "test_1222": _metrics(model, scaler, [item for item in test if item.get("is_target_player")]),
    }
    payload = {
        "schema_version": 1,
        "profile": "強化中AI2",
        "model_type": "masked_legal_action_mlp",
        "feature_names": feature_names,
        "scaler": {
            "mean": np.asarray(scaler.mean_, dtype=float).tolist(),
            "scale": np.asarray(scaler.scale_, dtype=float).tolist(),
        },
        "class_ids": [int(item) for item in model.classes_],
        "layers": [
            {
                "weights": np.asarray(weights, dtype=float).tolist(),
                "bias": np.asarray(bias, dtype=float).tolist(),
            }
            for weights, bias in zip(model.coefs_, model.intercepts_)
        ],
        "training": {
            "source_records": len(records),
            "train_records": len(train),
            "weighted_train_records": len(weighted_train),
            "target_repeat": int(target_repeat),
            "iterations": int(model.n_iter_),
            "loss": float(model.loss_),
            "hidden_layers": list(hidden_layers),
            "random_state": random_state,
            "selected_action_counts": dict(sorted(Counter(map(str, train_y)).items())),
        },
        "evaluation": evaluation,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    report = {
        key: value
        for key, value in payload.items()
        if key not in {"feature_names", "scaler", "class_ids", "layers"}
    }
    report["model_path"] = str(model_path)
    report["feature_count"] = len(feature_names)
    report["class_count"] = len(model.classes_)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--target-repeat", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=60)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = train_policy(
        args.data,
        args.model,
        args.report,
        target_repeat=max(1, int(args.target_repeat)),
        max_iter=max(1, int(args.max_iter)),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
