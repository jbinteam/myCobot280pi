#!/usr/bin/env python3
"""Streaming TCP bridge for this arm's Atom 6.5 and async UI commands.

Place beside pi_transport.py, with the original Server_280.py in this directory
or its parent. It retains the original serial settings and GPIO implementation.
"""
from pathlib import Path
import socket
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from Server_280 import MycobotServer, GPIO
from pi_transport import PacketBuffer, SerialTransport


class StreamingServer(MycobotServer):
    def connect(self):
        self.s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        transport = SerialTransport(self.mc)
        while True:
            conn,address = self.s.accept()
            self.logger.info('TCP connected: %s',address)
            with conn:
                conn.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                packets = PacketBuffer()
                try:
                    while True:
                        data = conn.recv(1024)
                        if not data:
                            break
                        packets.extend(data)
                        while True:
                            command = packets.pop()
                            if command is None:
                                break
                            started = time.monotonic()
                            genre = command[3]
                            if genre == 170:
                                GPIO.setmode(GPIO.BCM if command[4] == 0 else GPIO.BOARD)
                            elif genre == 171:
                                GPIO.setup(command[4],GPIO.OUT if command[5] else GPIO.IN)
                            elif genre == 172:
                                GPIO.output(command[4],command[5])
                            elif genre == 173:
                                conn.sendall(bytes([GPIO.input(command[4])]))
                                continue
                            reply = transport.exchange(command)
                            if reply:
                                conn.sendall(reply)
                            self.logger.info('command=0x%02x bytes=%s reply_bytes=%d elapsed_ms=%.2f',
                                             genre,command.hex(),len(reply),1000*(time.monotonic()-started))
                except Exception:
                    self.logger.exception('TCP client failed')
                finally:
                    # A lost connection must not leave the last reference moving.
                    transport.exchange(b'\xfe\xfe\x02\x29\xfa')
                    self.logger.info('TCP disconnected; Stop sent')


if __name__ == '__main__':
    host = sys.argv[1] if len(sys.argv) > 1 else '192.168.77.225'
    StreamingServer(host,9000,'/dev/ttyAMA0',1000000)
