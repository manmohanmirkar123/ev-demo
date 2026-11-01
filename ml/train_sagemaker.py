"""Train the days-to-limit model as a SageMaker training job (run against the SageMaker account).

Steps: build run-to-failure data -> S3 -> training job (XGBoost framework container, script mode,
entry point ml/sm_code/xgb_train.py) -> download model.json -> verify with the Lambda's pure-Python
evaluator. Publishing to the fleet account is a separate step (ml/publish_model.py), so the two
AWS accounts never need to trust each other.
"""
import argparse
import io
import json
import pathlib
import sys
import tarfile
import time

import boto3
from botocore.exceptions import ClientError

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "ml"), str(ROOT / "ml" / "sm_code"), str(ROOT / "lambda" / "scorer")]

from dataset import build_splits  # noqa: E402
from health import FEATURES  # noqa: E402
from tree_model import TreeModel  # noqa: E402
from hyperparams import HYPERPARAMS  # noqa: E402

# SageMaker XGBoost framework image registry accounts (image: sagemaker-xgboost:1.7-1).
XGBOOST_ACCOUNTS = {
    "us-east-1": "683313688378",
    "us-east-2": "257758044811",
    "us-west-2": "246618743249",
}
ROLE_NAME = "ev-pdm-sagemaker-exec"
MODEL_DIR = ROOT / "ml" / ".model"


def ensure_bucket(s3, name, region):
    try:
        s3.head_bucket(Bucket=name)
    except ClientError:
        kwargs = {} if region == "us-east-1" else {"CreateBucketConfiguration": {"LocationConstraint": region}}
        s3.create_bucket(Bucket=name, **kwargs)
        print(f"created bucket {name}")


def resolve_role(session, bucket):
    """Execution role for the training job, in order of preference:
    our own role (created if allowed) -> an existing role SageMaker can assume -> the role we run as (Studio).
    Console-only sandboxes often forbid iam:CreateRole but come with a SageMaker role already.
    """
    iam = session.client("iam")
    try:
        return ensure_role(iam, bucket)
    except ClientError as e:
        print(f"  can't create {ROLE_NAME} ({e.response['Error']['Code']}); looking for an existing SageMaker role")

    try:
        for page in iam.get_paginator("list_roles").paginate():
            for role in page["Roles"]:
                trust = json.dumps(role.get("AssumeRolePolicyDocument", {}))
                if "sagemaker.amazonaws.com" in trust and "AWSServiceRoleFor" not in role["RoleName"]:
                    print(f"  using existing role {role['RoleName']}")
                    return role["Arn"]
    except ClientError as e:
        print(f"  can't list roles ({e.response['Error']['Code']})")

    caller = session.client("sts").get_caller_identity()["Arn"]
    if ":assumed-role/" in caller:  # e.g. running in a SageMaker Studio terminal
        name = caller.split("/")[1]
        try:
            arn = iam.get_role(RoleName=name)["Role"]["Arn"]
        except ClientError:
            arn = f"arn:aws:iam::{caller.split(':')[4]}:role/{name}"
        print(f"  using the current role {name}")
        return arn
    sys.exit("No usable SageMaker execution role. In the IAM console find a role whose trusted entity is "
             "SageMaker and pass it with --role-arn arn:aws:iam::<account>:role/<name>")


def ensure_role(iam, bucket):
    try:
        return iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        pass
    arn = iam.create_role(
        RoleName=ROLE_NAME,
        AssumeRolePolicyDocument=json.dumps({
            "Version": "2012-10-17",
            "Statement": [{"Effect": "Allow", "Principal": {"Service": "sagemaker.amazonaws.com"},
                           "Action": "sts:AssumeRole"}],
        }),
    )["Role"]["Arn"]
    iam.put_role_policy(RoleName=ROLE_NAME, PolicyName="training", PolicyDocument=json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
             "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"]},
            {"Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents",
                                           "logs:DescribeLogStreams", "cloudwatch:PutMetricData",
                                           "ecr:GetAuthorizationToken", "ecr:BatchGetImage",
                                           "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"],
             "Resource": "*"},
        ],
    }))
    print(f"created role {ROLE_NAME}; waiting for IAM to propagate")
    time.sleep(15)
    return arn


def upload_code(s3, bucket, prefix):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in ("xgb_train.py", "hyperparams.py"):
            tar.add(ROOT / "ml" / "sm_code" / name, arcname=name)
    key = f"{prefix}/code/sourcedir.tar.gz"
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    return f"s3://{bucket}/{key}"


def wait(sm, job):
    last = None
    while True:
        desc = sm.describe_training_job(TrainingJobName=job)
        status = (desc["TrainingJobStatus"], desc.get("SecondaryStatus"))
        if status != last:
            print(f"  {time.strftime('%H:%M:%S')}  {status[0]} / {status[1]}")
            last = status
        if desc["TrainingJobStatus"] in ("Completed", "Failed", "Stopped"):
            return desc
        time.sleep(15)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--instance-type", default="ml.m5.large")
    ap.add_argument("--role-arn", help="existing SageMaker execution role (skip creating one)")
    ap.add_argument("--bucket", help="existing bucket (default: sagemaker-ev-pdm-<account>-<region>)")
    ap.add_argument("--image", help="override the XGBoost training image URI")
    ap.add_argument("--config", help="same-account mode: config.json from `make deploy SAGEMAKER=1`; uses the "
                                     "fleet's lake bucket + Terraform's SageMaker role and publishes the model")
    args = ap.parse_args()

    cfg = None
    if args.config:
        cfg = {k: v["value"] for k, v in json.loads(pathlib.Path(args.config).read_text()).items()}
        if not cfg.get("sagemaker_role_arn"):
            sys.exit("This deployment has no SageMaker role — redeploy with `make deploy SAGEMAKER=1`.")
        args.region = cfg["region"]

    session = boto3.Session(region_name=args.region)
    account = session.client("sts").get_caller_identity()["Account"]
    s3, sm = session.client("s3"), session.client("sagemaker")
    if cfg:
        bucket, prefix, role = cfg["lake_bucket"], "sagemaker", cfg["sagemaker_role_arn"]
        print(f"SageMaker in the fleet account {account}, region {args.region}")
    else:
        # "sagemaker" in the name: AWS's managed SageMaker policies only grant S3 access to such buckets.
        bucket = args.bucket or f"sagemaker-ev-pdm-{account}-{args.region}"
        prefix = "ev-pdm"
        print(f"SageMaker account {account}, region {args.region}")
        ensure_bucket(s3, bucket, args.region)
        role = args.role_arn or resolve_role(session, bucket)

    print("building run-to-failure training data from simulated fleet history…")
    train_csv, val_csv, n_tr, n_va = build_splits(ROOT / "ml" / ".data")
    for channel, path in (("train", train_csv), ("validation", val_csv)):
        s3.upload_file(str(path), bucket, f"{prefix}/data/{channel}/{path.name}")
    print(f"  {n_tr} train rows, {n_va} validation rows -> s3://{bucket}/{prefix}/data/")

    if args.image:
        image = args.image
    elif args.region in XGBOOST_ACCOUNTS:
        image = f"{XGBOOST_ACCOUNTS[args.region]}.dkr.ecr.{args.region}.amazonaws.com/sagemaker-xgboost:1.7-1"
    else:
        sys.exit(f"No XGBoost image mapping for {args.region}; pass --image")

    job = f"ev-pdm-rul-{time.strftime('%Y%m%d-%H%M%S')}"
    hyperparameters = {k: json.dumps(v) for k, v in HYPERPARAMS.items()}
    hyperparameters.update({
        "sagemaker_program": json.dumps("xgb_train.py"),
        "sagemaker_submit_directory": json.dumps(upload_code(s3, bucket, prefix)),
        "sagemaker_region": json.dumps(args.region),
    })
    channel = lambda name: {
        "ChannelName": name,
        "ContentType": "text/csv",
        "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3DataDistributionType": "FullyReplicated",
                                        "S3Uri": f"s3://{bucket}/{prefix}/data/{name}/"}},
    }
    sm.create_training_job(
        TrainingJobName=job,
        AlgorithmSpecification={
            "TrainingImage": image,
            "TrainingInputMode": "File",
            "MetricDefinitions": [
                {"Name": "validation:mae", "Regex": "validation-mae:([0-9\\.]+)"},
                {"Name": "validation:near_failure_mae", "Regex": "near-failure-mae:([0-9\\.]+)"},
            ],
        },
        RoleArn=role,
        HyperParameters=hyperparameters,
        InputDataConfig=[channel("train"), channel("validation")],
        OutputDataConfig={"S3OutputPath": f"s3://{bucket}/{prefix}/models/"},
        ResourceConfig={"InstanceType": args.instance_type, "InstanceCount": 1, "VolumeSizeInGB": 5},
        StoppingCondition={"MaxRuntimeInSeconds": 1800},
        Tags=[{"Key": "Project", "Value": "ev-pdm-poc"}],
    )
    print(f"training job {job} on {args.instance_type}")
    print(f"  console: https://{args.region}.console.aws.amazon.com/sagemaker/home?region={args.region}"
          f"#/jobs/{job}")
    desc = wait(sm, job)
    if desc["TrainingJobStatus"] != "Completed":
        sys.exit(f"training job {desc['TrainingJobStatus']}: {desc.get('FailureReason')}")

    for m in desc.get("FinalMetricDataList", []):
        print(f"  {m['MetricName']}: {m['Value']:.2f} days")
    if desc.get("BillableTimeInSeconds"):
        print(f"  billable time: {desc['BillableTimeInSeconds']}s")

    artifact = desc["ModelArtifacts"]["S3ModelArtifacts"]
    a_bucket, a_key = artifact[len("s3://"):].split("/", 1)
    with tarfile.open(fileobj=io.BytesIO(s3.get_object(Bucket=a_bucket, Key=a_key)["Body"].read())) as tar:
        model_json = tar.extractfile("model.json").read()

    tree = TreeModel(model_json)  # same evaluator the Lambda uses
    if tree.num_features != len(FEATURES):
        sys.exit(f"model has {tree.num_features} features, scorer expects {len(FEATURES)}")
    errors = []
    with open(val_csv) as f:
        for line in f:
            row = [float(x) for x in line.split(",")]
            errors.append(abs(tree.predict(row[1:]) - row[0]))
    print(f"  verified with pure-Python evaluator: validation MAE {sum(errors) / len(errors):.2f} days, "
          f"{len(tree.trees)} trees")

    info = {
        "trained_with": "sagemaker", "training_job": job, "account": account, "region": args.region,
        "artifact": artifact, "instance_type": args.instance_type,
        "metrics": {m["MetricName"]: m["Value"] for m in desc.get("FinalMetricDataList", [])},
        "features": FEATURES, "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "model.json").write_bytes(model_json)
    (MODEL_DIR / "model-info.json").write_text(json.dumps(info, indent=2))
    # One file to carry between accounts when the ML account is console-only (CloudShell / Studio download).
    bundle = {"info": info, "model": json.loads(model_json)}
    bundle_path = MODEL_DIR / "ev-pdm-model.json"
    bundle_path.write_text(json.dumps(bundle))
    print(f"saved {bundle_path}")
    if cfg:
        from publish_model import publish  # imported here: not shipped in the console-only bundle

        publish(s3, bucket, bundle)
    else:
        print("next: download it to your laptop, then `make publish-model FROM=<path to ev-pdm-model.json>`")


if __name__ == "__main__":
    main()
