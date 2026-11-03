#!/usr/bin/env python3
"""Find the --client-ip / --server-ip pair to use for Motive (NatNet).

Lists this machine's IPv4 addresses, then sends a NatNet "Ping" (NAT_CONNECT)
from each one as a broadcast to the command port.  Every Motive instance that
answers is printed together with the local address that reached it, i.e. the
exact values to pass as --client-ip and --server-ip.

    python find_mocap.py               # broadcast discovery
    python find_mocap.py --sweep       # also unicast-ping every host in each local /24
    python find_mocap.py --server-ip 192.168.31.51   # also ping one specific host

If nothing answers: make sure Motive is running with streaming enabled, the
cable/Wi-Fi is on the mocap network, and Windows Firewall allows Python
(UDP 1510/1511) on that network.
"""

from __future__ import annotations

import argparse
import ipaddress
import select
import socket
import struct
import time
from typing import Dict, List, Tuple

COMMAND_PORT = 1510
NAT_CONNECT = 0
NAT_SERVERINFO = 1


def local_ipv4_addresses() -> List[str]:
    """Best-effort list of this machine's IPv4 addresses (no extra deps)."""
    addrs = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addrs.add(info[4][0])
    except socket.gaierror:
        pass
    # The address used for the default route (no packet is actually sent).
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            addrs.add(s.getsockname()[0])
    except OSError:
        pass
    try:  # psutil gives the complete list if it happens to be installed
        import psutil  # type: ignore
        for nics in psutil.net_if_addrs().values():
            for a in nics:
                if a.family == socket.AF_INET:
                    addrs.add(a.address)
    except ImportError:
        pass
    return sorted(a for a in addrs if not a.startswith(("127.", "169.254.")))


def connect_packet() -> bytes:
    payload = bytearray(270)
    payload[0:4] = b"Ping"
    payload[265:269] = bytes([4, 2, 0, 0])  # client NatNet version
    body = bytes(payload) + b"\0"
    return struct.pack("<hh", NAT_CONNECT, len(body)) + body


def parse_server_info(data: bytes) -> Tuple[str, str, str]:
    payload = data[4:]
    name = payload[:256].split(b"\0", 1)[0].decode(errors="replace")
    app = ".".join(str(b) for b in payload[256:260])
    natnet = ".".join(str(b) for b in payload[260:264])
    return name, app, natnet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server-ip", action="append", default=[],
                        help="also ping this host directly (repeatable)")
    parser.add_argument("--sweep", action="store_true",
                        help="unicast-ping every address in each local /24")
    parser.add_argument("--timeout", type=float, default=2.0, help="seconds to wait for replies")
    args = parser.parse_args()

    locals_ = local_ipv4_addresses()
    print("This machine's IPv4 addresses (valid --client-ip values):")
    for ip in locals_ or ["(none found -- check `ipconfig`)"]:
        print(f"  {ip}")
    print()

    packet = connect_packet()
    socks: Dict[socket.socket, str] = {}
    for ip in locals_:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            s.bind((ip, 0))
        except OSError as exc:
            print(f"  could not bind {ip}: {exc}")
            s.close()
            continue
        net = ipaddress.ip_network(f"{ip}/24", strict=False)
        targets = {"255.255.255.255", str(net.broadcast_address), *args.server_ip}
        if args.sweep:
            targets.update(str(h) for h in net.hosts())
        for t in targets:
            try:
                s.sendto(packet, (t, COMMAND_PORT))
            except OSError:
                pass
        socks[s] = ip

    found: Dict[Tuple[str, str], Tuple[str, str, str]] = {}
    deadline = time.time() + args.timeout
    while socks and time.time() < deadline:
        ready, _, _ = select.select(list(socks), [], [], max(0.0, deadline - time.time()))
        for s in ready:
            try:
                data, (src, _) = s.recvfrom(65536)
            except OSError:
                continue
            if len(data) >= 4 and struct.unpack("<h", data[:2])[0] == NAT_SERVERINFO:
                found[(socks[s], src)] = parse_server_info(data)
    for s in socks:
        s.close()

    if not found:
        print("No NatNet servers answered.")
        print("  - Is Motive running with Streaming -> Broadcast Frame Data ON?")
        print("  - In Motive's streaming panel, check 'Local Interface' -- that is the --server-ip.")
        print("  - Allow python.exe through Windows Firewall on this network, or retry with --sweep.")
        return

    print("NatNet servers found:")
    for (client, server), (name, app, natnet) in sorted(found.items()):
        print(f"  {name} {app} (NatNet {natnet}) at {server}, reached from {client}")
        print(f"      --client-ip {client} --server-ip {server}")


if __name__ == "__main__":
    main()
