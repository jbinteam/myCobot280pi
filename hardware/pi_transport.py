"""Small packet/serial core shared by the Pi bridge and PC reply reader."""
import time

# Atom 6.5 on this arm does not acknowledge these commands. The UI sends them
# asynchronously and verifies mode changes with GET_FRESH_MODE afterwards.
# SET_ENCODERS_DRAG is also sent once asynchronously: do not hold subsequent
# telemetry behind a motor ACK that the PC does not consume.
NO_REPLY_COMMANDS = frozenset((0x16, 0x21, 0x22, 0x29, 0x30, 0x3E))


def matches_reply(genre, packet):
    """Match telemetry shape as well as its command code.

    GET_ANGLES carries six signed 16-bit angles. A short status/ACK packet
    with that code is not joint telemetry, even if its framing is valid.
    """
    return (len(packet) > 5 and packet[3] == genre
            and (genre not in (0x20, 0x3D) or len(packet) == 17))


class PacketBuffer:
    def __init__(self):
        self.data = bytearray()

    def extend(self, data):
        self.data.extend(data)

    def pop(self):
        while True:
            start = self.data.find(b'\xfe\xfe')
            if start < 0:
                self.data[:] = self.data[-1:] if self.data[-1:] == b'\xfe' else b''
                return None
            del self.data[:start]
            if len(self.data) < 3:
                return None
            length = self.data[2]
            # Servo status/debug traffic can end in FE immediately before a
            # reply's FE FE header. None of this bridge's supported packets
            # has a 254-byte length. Slide across overlapping sync markers
            # instead of waiting for a spurious 257-byte packet.
            if length == 0xfe:
                del self.data[0]
                continue
            if length < 2:
                del self.data[0]
                continue
            size = length+3
            if len(self.data) < size:
                return None
            if self.data[size-1] != 0xfa:
                del self.data[0]
                continue
            packet = bytes(self.data[:size])
            del self.data[:size]
            return packet


class SerialTransport:
    def __init__(self, serial, reply_timeout=.15):
        self.serial = serial
        self.reply_timeout = reply_timeout
        self.buffer = PacketBuffer()

    def exchange(self, command):
        if (len(command) < 5 or command[:2] != b'\xfe\xfe'
                or len(command) != command[2]+3 or command[-1] != 0xfa):
            raise ValueError('Invalid robot command packet')
        if self.serial.write(command) != len(command):
            raise ConnectionError('Incomplete serial command write')
        self.serial.flush()
        if command[3] in NO_REPLY_COMMANDS:
            return b''
        deadline = time.monotonic()+self.reply_timeout
        while time.monotonic() < deadline:
            packet = self.buffer.pop()
            if packet is not None:
                if packet != command and matches_reply(command[3], packet):
                    return packet
                continue  # Discard echoes and delayed ACKs from other commands.
            self.serial.timeout = min(.02,max(0.,deadline-time.monotonic()))
            self.buffer.extend(self.serial.read(max(1,self.serial.in_waiting)))
        return b''
