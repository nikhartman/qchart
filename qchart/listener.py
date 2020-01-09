import time
import zmq
import json
import subprocess
import sys
from pathlib import Path, PurePath
from qchart.config import config


def listener_is_running(timeout=None):

    addr = config["network"]["addr"]
    port = config["network"]["port"]
    srvr = f"tcp://{addr}:{port}"

    if timeout is None:
        timeout = config["client"]["send_timeout"]

    context = zmq.Context()
    context.setsockopt(zmq.LINGER, timeout)
    socket = context.socket(zmq.PUSH)
    socket.connect(srvr)

    enc_data = json.dumps({"ping": "pong"}).encode()
    socket.send(enc_data)

    t0 = time.time()
    socket.close()
    context.term()

    if (time.time() - t0) > (timeout / 1000.0):
        return False

    return True


def start_listener():

    if listener_is_running():
        print("qchart listener is already running.")
        return

    python_path = str(Path(sys.executable))

    import qchart
    qchart_path_parts = PurePath(qchart.__file__).parts[:-2]
    qchart_path_full = str(Path(*qchart_path_parts, "listener_start.py"))

    print(f"starting qchart listener at {qchart_path_full} ...")
    subprocess.Popen(
        [python_path, qchart_path_full],
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    time.sleep(1.0)

    for _ in range(5):
        if listener_is_running():
            print("qchart listener successfully started.")
            return
        time.sleep(1.0)

    raise RuntimeWarning("Failed to start listener!")
