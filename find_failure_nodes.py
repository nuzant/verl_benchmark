import socket

nodes_to_ip = {}
for i in range(200):
    try:
        ip = socket.gethostbyname(f"slurmd-{i}")
        nodes_to_ip[i] = ip
        print(f"slurmd-{i}: {ip}")
    except:
        pass

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ip", required=True, type=str)

args = parser.parse_args()
for k, v in nodes_to_ip.items():
    if v == args.ip:
        print(f"ip = {args.ip} node = slurmd-{k}")