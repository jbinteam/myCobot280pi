import socket
import struct
import threading
import time
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from pymycobot import MyCobot280Socket
from pymycobot.generate import CommandGenerator
from hardware.pi_transport import PacketBuffer, SerialTransport, NO_REPLY_COMMANDS
from webui.robot_transport import RobotReplyReader, reply_timeout


def packet(genre,payload=b''):
    return b'\xfe\xfe'+bytes((len(payload)+2,genre))+payload+b'\xfa'


class FakeSerial:
    def __init__(self,chunks=()):
        self.chunks=list(chunks);self.writes=[];self.reads=0;self.timeout=.1
    def write(self,data):self.writes.append(data);return len(data)
    def flush(self):pass
    @property
    def in_waiting(self):return len(self.chunks[0]) if self.chunks else 0
    def read(self,size):
        self.reads+=1
        return self.chunks.pop(0) if self.chunks else b''


class FakeSocket:
    def __init__(self,chunks=(),delay=0):
        self.chunks=list(chunks);self.writes=[];self.timeouts=[];self.delay=delay
    def sendall(self,data):self.writes.append(data)
    def fileno(self):return 1
    def settimeout(self,timeout):self.timeouts.append(timeout)
    def recv(self,size):
        if self.delay:
            if self.timeouts[-1]<self.delay:raise socket.timeout()
            time.sleep(self.delay);self.delay=0
        chunk = self.chunks.pop(0) if self.chunks else b''
        if isinstance(chunk,Exception):raise chunk
        return chunk


class RobotTransportTest(unittest.TestCase):
    def robot(self,chunks=(),delay=0):
        robot=MyCobot280Socket.__new__(MyCobot280Socket)
        CommandGenerator.__init__(robot,False)
        robot.sock=FakeSocket(chunks,delay);robot.lock=threading.Lock()
        robot._read=RobotReplyReader(robot)
        return robot

    def test_packets_can_be_split_at_every_byte_and_combined(self):
        first=packet(0x22,b'\xfa\xfe\xfe\x00')
        second=packet(0x20,b'\x00'*12)
        buffer=PacketBuffer();found=[]
        for byte in b'noise'+first+second:
            buffer.extend(bytes([byte]))
            while True:
                result=buffer.pop()
                if result is None:break
                found.append(result)
        self.assertEqual(found,[first,second])
        buffer.extend(first+second)
        self.assertEqual(buffer.pop(),first);self.assertEqual(buffer.pop(),second)

    def test_debug_checksum_overlapping_header_preserves_split_reply(self):
        reply = packet(0x2b, b'\x00')
        for prefix in (b'\xfe', b'\xfe\xfe', b'\xff\xff\x07\x02\x01\xf5\xfe'):
            for split in range(len(prefix+reply)+1):
                with self.subTest(prefix=prefix, split=split):
                    buffer = PacketBuffer()
                    data = prefix+reply
                    buffer.extend(data[:split])
                    first = buffer.pop()
                    buffer.extend(data[split:])
                    self.assertEqual(first if first is not None else buffer.pop(), reply)
                    self.assertIsNone(buffer.pop())

    def test_captured_arm_debug_bytes_do_not_hide_angle_or_motion_reply(self):
        import json
        capture = Path(__file__).resolve().parents[2]/'output/pi_serial_readonly_diagnostic.txt'
        if not capture.exists():
            self.skipTest('Captured serial fixture unavailable')
        for line in capture.read_text().splitlines():
            row = json.loads(line)
            genre = int(row['genre'], 16)
            buffer = PacketBuffer(); buffer.extend(bytes.fromhex(row['raw_hex']))
            reply = buffer.pop()
            self.assertIsNotNone(reply)
            self.assertEqual(reply[3], genre)
            self.assertEqual(len(reply), 17 if genre == 0x20 else 6)

    def test_bad_packet_footer_and_length_resynchronize(self):
        good=packet(0x17,b'\x01');buffer=PacketBuffer()
        buffer.extend(b'\xfe\xfe\x00\xfe\xfe\x02\x20\x00'+good)
        self.assertEqual(buffer.pop(),good)

    def test_unacknowledged_motor_and_mode_commands_never_read_serial(self):
        serial=FakeSerial();transport=SerialTransport(serial)
        for genre in NO_REPLY_COMMANDS:
            self.assertEqual(transport.exchange(packet(genre)),b'')
        self.assertEqual(serial.reads,0)
        self.assertEqual(len(serial.writes),len(NO_REPLY_COMMANDS))

    def test_serial_query_discards_echo_and_motor_ack_without_losing_reply(self):
        query=packet(0x20);reply=packet(0x20,b'\x00'*12)
        incoming=query+packet(0x22,b'\x01')+reply
        serial=FakeSerial([incoming[:7],incoming[7:]])
        self.assertEqual(SerialTransport(serial).exchange(query),reply)
        self.assertEqual(serial.writes,[query])

    def test_combined_motor_and_query_frames_are_written_separately(self):
        motor=packet(0x22,b'\x00'*12+b'\x0a');query=packet(0x20)
        serial=FakeSerial([packet(0x20,b'\x00'*12)])
        buffer=PacketBuffer();buffer.extend(motor+query);transport=SerialTransport(serial)
        self.assertEqual(transport.exchange(buffer.pop()),b'')
        self.assertTrue(transport.exchange(buffer.pop()))
        self.assertEqual(serial.writes,[motor,query])

    def test_real_sdk_handles_split_reply_and_coalesced_ack_without_resending(self):
        expected=[-10.1,-.43,-.43,-.08,.26,-.26]
        reply=packet(0x20,struct.pack('>6h',*[round(x*100) for x in expected]))
        incoming=packet(0x22,b'\x01')+reply
        robot=self.robot([incoming[:9],incoming[9:]])
        self.assertEqual(robot.get_angles(),expected)
        self.assertEqual(robot.sock.writes,[packet(0x20)])

    def test_sdk_skips_same_command_short_replies_and_keeps_six_joint_packet(self):
        expected=[1,2,3,4,5,6]
        reply=packet(0x20,struct.pack('>6h',*[x*100 for x in expected]))
        for size in (1,2,10,14):
            with self.subTest(size=size):
                robot=self.robot([packet(0x20,b'\0'*size)+reply])
                self.assertEqual(robot.get_angles(),expected)
                self.assertEqual(robot.sock.writes,[packet(0x20)])

    def test_serial_skips_malformed_angle_reply_without_resending(self):
        reply=packet(0x20,b'\0'*12)
        serial=FakeSerial([packet(0x20,b'\1')+packet(0x20,b'\0'*10)+reply])
        self.assertEqual(SerialTransport(serial).exchange(packet(0x20)),reply)
        self.assertEqual(serial.writes,[packet(0x20)])

    def test_sdk_timeout_diagnostics_distinguish_missing_and_malformed_replies(self):
        for malformed in (False,True):
            with self.subTest(malformed=malformed):
                chunks=[]
                for _ in range(3):
                    if malformed:chunks.append(packet(0x20,b'\1'))
                    chunks.append(socket.timeout())
                robot=self.robot(chunks)
                self.assertEqual(robot.get_angles(),-1)
                detail=robot._read.failure_detail(0x20)
                self.assertIn('3 failed reply attempt',detail)
                self.assertIn('800 ms',detail)
                self.assertIn('malformed payload sizes [1, 1, 1]' if malformed else 'no complete matching reply',detail)
                self.assertEqual(len(robot.sock.writes),3)

    def test_streaming_reply_budget_bounds_retries_and_restores_idle_timeout(self):
        robot = self.robot([socket.timeout(), socket.timeout(), packet(0x51, b'\x01')])
        with reply_timeout(robot, .25):
            self.assertEqual(robot.is_all_servo_enable(), 1)
        self.assertEqual(robot._read.timeout, .8)
        queries = list(robot._read.recent_queries)
        self.assertEqual([q['outcome'] for q in queries], ['timeout', 'timeout', 'reply'])
        self.assertTrue(all(q['timeout_s'] == .25 for q in queries))
        self.assertTrue(all(q['elapsed_s'] >= 0 for q in queries))
        self.assertEqual(len(robot.sock.writes), 3)
        with self.assertRaises(RuntimeError):
            with reply_timeout(robot, .25):
                raise RuntimeError('cancelled')
        self.assertEqual(robot._read.timeout, .8)

    def test_pi_query_wait_is_shorter_than_streaming_pc_wait(self):
        serial = FakeSerial()
        self.assertEqual(SerialTransport(serial).reply_timeout, .15)

    def test_total_query_budget_bounds_sdk_retries_and_restores_reader(self):
        robot = self.robot()
        def no_reply(size):
            time.sleep(robot.sock.timeouts[-1])
            raise socket.timeout()
        robot.sock.recv = no_reply
        began=time.monotonic()
        with self.assertRaisesRegex(TimeoutError, 'budget exhausted'):
            with reply_timeout(robot, .03, budget_s=.04):
                robot.get_angles()
        self.assertLess(time.monotonic()-began, .15)
        self.assertEqual(robot._read.timeout,.8)
        self.assertIsNone(robot._read.deadline)
        self.assertEqual(robot._read.recent_queries[-1]['outcome'],'budget_exhausted')
        robot.sock.recv=lambda size:packet(0x20,b'\0'*12)
        self.assertEqual(robot.get_angles(),[0]*6)

    def test_total_budget_is_shared_by_consecutive_health_queries(self):
        robot=self.robot()
        with reply_timeout(robot,.25,budget_s=.02):
            deadline=robot._read.deadline
            with reply_timeout(robot,.25,budget_s=1.):
                self.assertEqual(robot._read.deadline,deadline)
            time.sleep(.03)
            with self.assertRaises(TimeoutError):robot.is_power_on()
        self.assertIsNone(robot._read.deadline)

    def test_legacy_half_second_delay_does_not_resend_the_query(self):
        robot=self.robot([packet(0x17,b'\x01')],delay=.55)
        self.assertEqual(robot.get_fresh_mode(),1)
        self.assertEqual(robot.sock.writes,[packet(0x17)])

    def test_real_sdk_async_motor_target_sends_once_without_reading(self):
        robot=self.robot()
        robot.send_angles([1,0,0,0,0,0],10,_async=True)
        self.assertEqual(len(robot.sock.writes),1)
        self.assertEqual(robot.sock.timeouts,[])
        self.assertEqual(robot.sock.writes[0][3],0x22)

    def test_encoder_drag_packet_contains_all_six_targets_and_speeds_sent_once(self):
        from webui.encoder_drag import send_once
        from pymycobot.error import MyCobot280DataException
        robot=self.robot()
        encoders=[2048,1900,1600,2200,2000,2100];speeds=[1,20,30,40,50,60]
        send_once(robot,encoders,speeds)
        self.assertEqual(robot.sock.writes,[packet(0x3E,struct.pack('>12h',*encoders,*speeds))])
        self.assertEqual(robot.sock.timeouts,[])
        with self.assertRaises(MyCobot280DataException):send_once(robot,encoders,[-1]*6)
        self.assertEqual(len(robot.sock.writes),1)

    def test_encoder_speed_send_does_not_delay_following_pi_telemetry(self):
        motor=packet(0x3E,struct.pack('>12h',*([2048]*6+[12]*6)))
        query=packet(0x20); reply=packet(0x20,b'\0'*12)
        serial=FakeSerial([packet(0x3E,b'\1')+reply])
        transport=SerialTransport(serial)
        self.assertEqual(transport.exchange(motor),b'')
        self.assertEqual(serial.reads,0)
        self.assertEqual(transport.exchange(query),reply)
        self.assertEqual(serial.writes,[motor,query])

    def test_encoder_query_skips_short_status_reply_without_resend(self):
        expected=[2048,1900,1600,2200,2000,2100]
        robot=self.robot([packet(0x3D,b'\1')+packet(0x3D,struct.pack('>6h',*expected))])
        self.assertEqual(robot.get_encoders(),expected)
        self.assertEqual(robot.sock.writes,[packet(0x3D)])

    def test_socket_disconnect_is_reported_immediately(self):
        robot=self.robot()
        with self.assertRaises(ConnectionError):robot.get_angles()
        self.assertEqual(len(robot.sock.writes),1)

    def test_pi_server_parses_combined_commands_and_stops_on_disconnect(self):
        from hardware import pi_transport
        legacy=SimpleNamespace(MycobotServer=object,GPIO=Mock())
        spec=importlib.util.spec_from_file_location('tested_pi_server',Path(__file__).resolve().parents[2]/'hardware/pi_server.py')
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'Server_280':legacy,'pi_transport':pi_transport}):
            spec.loader.exec_module(module)
        motor=packet(0x22,b'\x00'*12+b'\x0a');query=packet(0x17)
        reply=packet(0x17,b'\x01');data=motor+query
        conn=Mock();conn.recv.side_effect=[data[:2],data[2:],b'']
        conn.__enter__=Mock(return_value=conn);conn.__exit__=Mock(return_value=False)
        listener=Mock();listener.accept.side_effect=[(conn,('127.0.0.1',1234)),StopIteration]
        server=module.StreamingServer();server.s=listener;server.mc=FakeSerial([reply]);server.logger=Mock()
        with self.assertRaises(StopIteration):server.connect()
        self.assertEqual(server.mc.writes,[motor,query,packet(0x29)])
        conn.sendall.assert_called_once_with(reply)


if __name__=='__main__':unittest.main()
