"""Publish synthetic fleet telemetry to AWS IoT Core over MQTT (X.509 certs from Terraform).

Each tick = one simulated day for every vehicle, so 120 days of fleet life plays out in ~2 minutes.
"""
import argparse
import json
import pathlib
import sys
import time

from fleet import make_fleet

ROOT = pathlib.Path(__file__).resolve().parent.parent
CERTS = ROOT / "simulator" / "certs"


def load_config():
    path = ROOT / "config.json"
    if not path.exists():
        sys.exit("config.json not found — run `make deploy` first.")
    return {k: v["value"] for k, v in json.loads(path.read_text()).items()}


def connect(endpoint, timeout=15):
    """Connect and block until IoT Core's CONNACK. AWS IoT drops a client that publishes before the
    connection is acknowledged, and paho's automatic resend after reconnecting repeats the drop."""
    import threading

    import paho.mqtt.client as mqtt

    connected = threading.Event()

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            print(f"  connect refused: {reason_code}", flush=True)
        else:
            connected.set()

    def on_disconnect(client, userdata, flags, reason_code, properties):
        connected.clear()
        if reason_code != 0:
            print(f"  disconnected by IoT Core: {reason_code} — reconnecting", flush=True)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"ev-sim-{int(time.time())}")
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.tls_set(ca_certs=str(CERTS / "AmazonRootCA1.pem"),
                   certfile=str(CERTS / "device.pem.crt"),
                   keyfile=str(CERTS / "private.pem.key"))
    client.connect(endpoint, 8883)
    client.loop_start()
    if not connected.wait(timeout):
        client.loop_stop()
        sys.exit(f"No CONNACK from {endpoint} within {timeout}s — check the certificate and IoT policy "
                 "(re-run `make deploy` after a sandbox reset).")
    client.connected = connected
    return client


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vehicles", type=int, default=20)
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--tick", type=float, default=1.0, help="seconds per simulated day")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="print messages instead of publishing")
    args = ap.parse_args()

    fleet = make_fleet(args.vehicles, args.seed)
    client = None
    if not args.dry_run:
        cfg = load_config()
        client = connect(cfg["iot_endpoint"])
        print(f"Connected to {cfg['iot_endpoint']}")

    pending = []
    run_id = time.strftime("%Y%m%dT%H%M%S")  # lets the scorer tell a replay from a late message
    try:
        for day in range(args.days):
            if client and not client.connected.wait(30):
                sys.exit("Lost the IoT Core connection and couldn't reconnect within 30s.")
            with_dtc = []
            for v in fleet:
                msg = {**v.telemetry(day), "run_id": run_id}
                topic = f"ev/telemetry/{v.vehicle_id}"
                if msg["dtc"]:
                    with_dtc.append(f"{v.vehicle_id}:{','.join(msg['dtc'])}")
                if client:
                    pending.append(client.publish(topic, json.dumps(msg), qos=1))
                elif day == 0:
                    print(topic, json.dumps(msg))
            if client:
                pending = [p for p in pending if not p.is_published()]
            # Today's batch is normally still in flight; only flag a backlog that builds across days.
            backlog = f"  unacked backlog: {len(pending)}" if len(pending) > len(fleet) else ""
            print(f"day {day:3d}  {len(fleet)} vehicles  DTCs: {' '.join(with_dtc) or '-'}{backlog}")
            time.sleep(0 if args.dry_run else args.tick)
    finally:
        if client:
            deadline = time.time() + 30  # wait for IoT Core to acknowledge every QoS1 message
            for p in pending:
                try:
                    p.wait_for_publish(max(0.1, deadline - time.time()))
                except (RuntimeError, ValueError):
                    pass
            lost = sum(1 for p in pending if not p.is_published())
            if lost:
                print(f"warning: {lost} messages were not acknowledged by IoT Core")
            client.loop_stop()
            client.disconnect()


if __name__ == "__main__":
    main()
