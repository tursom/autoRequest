import asyncio
import json
import sys
import time
from socket import *




async def main():
    s = socket(AF_INET, SOCK_DGRAM)
    s.setsockopt(SOL_SOCKET, SO_BROADCAST, 1)
    s.bind(('', 12345))
    while True:
        data, address = s.recvfrom(65535)
        print('Server received from {}:{}'.format(address, data.decode('utf-8')))


if __name__ == '__main__':
    asyncio.run(main())
