import asyncio
import json
import sys
import time
from socket import *


def send_broadcast(sock, config) -> bool:
    host = '<broadcast>'
    port = 12345
    text = f"{time.asctime(time.localtime(time.time()))} broadcast"
    if "host" in config:
        host = config["host"]
    if "text" in config:
        text = config["text"]
    addr = (host, port)
    sock.sendto(text.encode(), addr)
    return True


async def loop(loop_config, sock):
    delay = float(loop_config["delay"])
    times = -1
    if "timeunit" in loop_config:
        timeunit = loop_config["timeunit"]
        if timeunit == "minutes":
            delay = delay * 60
        elif timeunit == "hours":
            delay = delay * 60 * 60
        elif timeunit == "days":
            delay = delay * 60 * 60 * 24
        else:
            print(f"unknown timeunit {timeunit}")
    if "times" in loop_config:
        times = int(loop_config["times"])
    if times >= 0:
        times -= 1
        while times > 0:
            times -= 1
            await asyncio.sleep(delay)
            if not send_broadcast(sock, loop_config):
                times += 1
    else:
        while True:
            await asyncio.sleep(delay)
            send_broadcast(sock, loop_config)


async def main():
    handle_file = []
    tasks = []

    udpCliSock = socket(AF_INET, SOCK_DGRAM)
    udpCliSock.bind(('', 0))
    udpCliSock.setsockopt(SOL_SOCKET, SO_BROADCAST, 1)

    for arg in sys.argv[1:]:
        handle_file.append(arg)

    for file_name in handle_file:
        file_open = open(file_name, "rb")
        text = file_open.read().decode()
        config_list = json.loads(text)
        for config in config_list:
            send_broadcast(udpCliSock, config)
            task = asyncio.create_task(loop(config, udpCliSock))
            tasks.append(task)
    for task in tasks:
        await task


if __name__ == '__main__':
    asyncio.run(main())
