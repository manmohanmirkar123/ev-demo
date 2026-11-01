"""Pure-Python inference for an XGBoost JSON model (regression, gbtree).

Lets the Lambda run the SageMaker-trained model with no xgboost/numpy dependency and no endpoint,
which also means the model can be trained in one AWS account and served from another.
"""
import json
import struct


def _f32(x):
    # XGBoost compares features as float32; match it so predictions agree exactly at split boundaries.
    return struct.unpack("f", struct.pack("f", x))[0]


class TreeModel:
    def __init__(self, model):
        m = json.loads(model) if isinstance(model, (str, bytes, bytearray)) else model
        learner = m["learner"]
        # "6.6E1" in XGBoost 1.x, "[6.6E1]" in 2.x+
        self.base_score = float(str(learner["learner_model_param"]["base_score"]).strip("[]"))
        self.num_features = int(learner["learner_model_param"]["num_feature"])
        # JSON prints float32 thresholds as short decimals; round back to float32 or values that sit
        # exactly on a split boundary take the other branch.
        self.trees = [
            (t["left_children"], t["right_children"], t["split_indices"],
             [_f32(c) for c in t["split_conditions"]], t["default_left"])
            for t in learner["gradient_booster"]["model"]["trees"]
        ]

    def predict(self, x):
        if len(x) != self.num_features:
            raise ValueError(f"model expects {self.num_features} features, got {len(x)}")
        x = [_f32(v) for v in x]
        total = self.base_score
        for left, right, index, cond, default_left in self.trees:
            node = 0
            while left[node] != -1:
                value = x[index[node]]
                if value != value:  # NaN -> learned default direction
                    node = left[node] if default_left[node] else right[node]
                else:
                    node = left[node] if value < cond[node] else right[node]
            total += cond[node]  # leaf value is stored in split_conditions
        return total
