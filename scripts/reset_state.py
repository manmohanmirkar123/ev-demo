"""Clear the health and work-order tables so the demo can be replayed from day 0."""
import json
import pathlib

import boto3

ROOT = pathlib.Path(__file__).resolve().parent.parent
cfg = {k: v["value"] for k, v in json.loads((ROOT / "config.json").read_text()).items()}
ddb = boto3.Session(region_name=cfg["region"]).resource("dynamodb")

for name, keys in ((cfg["health_table"], ("vehicle_id", "component")),
                   (cfg["work_orders_table"], ("work_order_id",))):
    table = ddb.Table(name)
    deleted, kwargs = 0, {"ProjectionExpression": ", ".join(f"#k{i}" for i in range(len(keys))),
                          "ExpressionAttributeNames": {f"#k{i}": k for i, k in enumerate(keys)}}
    with table.batch_writer() as writer:
        while True:
            page = table.scan(**kwargs)
            for item in page["Items"]:
                writer.delete_item(Key={k: item[k] for k in keys})
                deleted += 1
            if "LastEvaluatedKey" not in page:
                break
            kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]
    print(f"{name}: deleted {deleted} items")
