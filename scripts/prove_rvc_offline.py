"""Run RVC inference while rejecting every Python socket creation."""

import os
import runpy
import socket
import ssl  # load SSL's socket subclasses before replacing socket.socket
import sys
from pathlib import Path


class BlockedSocket(socket.socket):
    def __new__(cls, *args, **kwargs):
        raise RuntimeError("Network socket blocked during local RVC inference")


socket.socket = BlockedSocket
socket.create_connection = BlockedSocket
root = Path(__file__).resolve().parent.parent / "external" / "RVC"
os.chdir(root)
sys.path.insert(0, str(root))
sys.argv = ["infer.cli", *sys.argv[1:]]
runpy.run_module("infer.cli", run_name="__main__")
