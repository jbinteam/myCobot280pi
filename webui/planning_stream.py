"""Bounded batches of read-only RRT* diagnostics for previews and ROS clients."""
import json
from time import monotonic


class PlanningStream:
    def __init__(self, write):
        self.write = write
        self.pending = []
        self.last_send = monotonic()
        self.closed = False

    def send(self, message):
        if self.closed:
            return
        try:
            self.write((json.dumps(message, allow_nan=False)+'\n').encode())
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.closed = True

    def flush(self):
        if self.pending:
            self.send({'type':'rrt', 'events':self.pending})
            self.pending = []
            self.last_send = monotonic()

    def observe(self, event):
        if self.closed:
            return
        self.pending.append(event)
        if event['kind'] in ('begin', 'end') or len(self.pending)>=32 or monotonic()-self.last_send>=.1:
            self.flush()

    def result(self, value):
        self.flush()
        self.send({'type':'result', 'result':value})

    def error(self, error, code=503):
        self.flush()
        self.send({'type':'error', 'error':str(error), 'code':code})
