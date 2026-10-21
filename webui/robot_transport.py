"""Frame TCP replies without assuming one recv call equals one robot packet."""
import socket
import time
from collections import deque
from contextlib import contextmanager
from pymycobot import MyCobot280Socket
from hardware.pi_transport import PacketBuffer, matches_reply

REPLY_TIMEOUT_SECONDS = .8


class RobotReplyReader:
    def __init__(self, robot, timeout=REPLY_TIMEOUT_SECONDS):
        self.robot, self.timeout = robot, timeout
        self.deadline = None
        self.buffer = PacketBuffer()
        self.recent_queries = deque(maxlen=64)

    def failure_detail(self, genre):
        queries = [q for q in self.recent_queries if q['genre'] == genre][-3:]
        if not queries or queries[-1]['outcome'] == 'reply':
            return ''
        rejected = [size for query in queries for size in query['rejected_payload_bytes']]
        detail = f'query 0x{genre:02x}: {len(queries)} failed reply attempt(s), {self.timeout*1000:g} ms timeout each'
        if rejected:
            detail += f'; discarded malformed payload sizes {rejected} (expected 12 bytes for six joint angles)'
        else:
            detail += '; no complete matching reply received'
        return detail

    def __call__(self, genre, method=None, **kwargs):
        if method != 'socket':
            raise ValueError('Robot reply reader accepts TCP packets only')
        query = {'genre': genre, 'outcome': 'waiting', 'rejected_payload_bytes': [],
                 'started_at': time.monotonic(), 'timeout_s': self.timeout}
        self.recent_queries.append(query)
        deadline = time.monotonic()+self.timeout
        if self.deadline is not None:
            deadline = min(deadline, self.deadline)
            if deadline <= time.monotonic():
                query.update(outcome='budget_exhausted', elapsed_s=time.monotonic()-query['started_at'])
                raise TimeoutError('Robot telemetry query budget exhausted')
        while True:
            packet = self.buffer.pop()
            if packet is not None:
                # Async command ACKs, echoed queries and unrelated replies must
                # not discard a following telemetry frame or trigger a resend.
                if matches_reply(genre, packet):
                    query['outcome'] = 'reply'
                    query['elapsed_s'] = time.monotonic()-query['started_at']
                    return packet
                if packet[3] == genre and len(packet) > 5:
                    query['rejected_payload_bytes'].append(len(packet)-5)
                continue
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                query['outcome'] = 'timeout'
                query['elapsed_s'] = time.monotonic()-query['started_at']
                return b''
            self.robot.sock.settimeout(remaining)
            try:
                data = self.robot.sock.recv(1024)
            except socket.timeout:
                query['outcome'] = 'timeout'
                query['elapsed_s'] = time.monotonic()-query['started_at']
                return b''
            if not data:
                query['outcome'] = 'closed'
                query['elapsed_s'] = time.monotonic()-query['started_at']
                raise ConnectionError('Robot TCP connection closed')
            self.buffer.extend(data)



@contextmanager
def reply_timeout(robot, timeout, *, budget_s=None):
    """Bound SDK retries during streaming; caller holds the controller lock."""
    reader = getattr(robot, '_read', None)
    if not isinstance(reader, RobotReplyReader):
        yield
        return
    previous, previous_deadline = reader.timeout, reader.deadline
    reader.timeout = timeout
    if budget_s is not None:
        deadline = time.monotonic()+budget_s
        reader.deadline = min(previous_deadline, deadline) if previous_deadline is not None else deadline
    try:
        yield
    finally:
        reader.timeout = previous
        reader.deadline = previous_deadline


def connect_robot(host, port):
    robot = MyCobot280Socket(host, port)
    # Keep the SDK's class name, packet encoder, reply decoder and command
    # validation. Only replace its one-recv TCP reader on this connection.
    robot._read = RobotReplyReader(robot)
    robot.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return robot
