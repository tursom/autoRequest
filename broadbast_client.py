import time
from socket import *


def main():
    PORT = 12345
    BUFSIZE = 1024
    udp_sock = socket(AF_INET, SOCK_DGRAM)
    udp_sock.bind(('', PORT))
    print('wating for message...')
    while True:
        data, addr = udp_sock.recvfrom(BUFSIZE)
        print(f'...{time.asctime(time.localtime(time.time()))} received ->{addr}  {data.decode()}')


if __name__ == '__main__':
    main()
