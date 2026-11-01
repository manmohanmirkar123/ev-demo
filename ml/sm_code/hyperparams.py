"""XGBoost hyperparameters. No dependencies, so the job launcher can import it where numpy isn't installed
(CloudShell); xgb_train.py uses the same values inside the training container."""

HYPERPARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "min_child_weight": 5,
    "num_round": 400,
    "early_stopping_rounds": 30,
}
