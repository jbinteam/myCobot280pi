"""Bounded read-only RRT diagnostics shared by the Python and C++ frontends.

Workers update joint-space data only. ROS timers serialize at most 10 Hz; FK
and marker generation belong to the separate visualization process.
"""
from collections import OrderedDict
import copy
import json
import threading
from time import monotonic


class PlanningVisualization:
    max_nodes = 4096
    max_attempts = 16

    def __init__(self):
        self.lock = threading.Lock()
        self.enabled = False
        self.epoch = 0
        self.dirty = False
        self.last_send = 0.
        self._clear()

    def _clear(self):
        self.nodes = []
        self.route = []
        self.goal = None
        self.attempts = OrderedDict()
        self.attempt = self.rewires = 0
        self.status = 'idle'

    def set_enabled(self, enabled):
        with self.lock:
            if bool(enabled) != self.enabled:
                self.enabled = bool(enabled)
                self.epoch += 1
                self._clear()
                self.dirty = True

    def begin(self):
        with self.lock:
            if not self.enabled:
                return None
            self.epoch += 1
            epoch = self.epoch
            self._clear()
            self.status = 'planning'
            self.dirty = True
        def observe(event):
            # Diagnostic failures must never change a motion decision.
            try:
                self.observe(epoch, event)
            except (KeyError, TypeError, ValueError, IndexError):
                pass
        return epoch, observe

    def observe(self, epoch, event):
        with self.lock:
            if not self.enabled or epoch != self.epoch:
                return
            kind = event['kind']
            if kind == 'begin':
                self.goal = tuple(event['goal_deg'])
                self.nodes = [{'q_deg':event['start_deg'], 'parent':-1}]
                self.route = []
                self.attempt += 1
                self.rewires = 0
                self.status = 'searching'
            elif kind == 'add':
                index = event['index']
                if index == len(self.nodes) and index < self.max_nodes:
                    self.nodes.append({'q_deg':event['q_deg'], 'parent':event['parent']})
            elif kind == 'rewire':
                if 0 <= event['index'] < len(self.nodes):
                    self.nodes[event['index']]['parent'] = event['parent']
                    self.rewires += 1
            elif kind == 'end':
                self.status = event['status']
                self.rewires = event['rewires']
                if self.status == 'direct_optimal' and event.get('path_deg'):
                    self.nodes = [{'q_deg':q, 'parent':i-1} for i,q in enumerate(event['path_deg'][:self.max_nodes])]
                if self.goal is not None:
                    self.attempts[self.goal] = (copy.deepcopy(self.nodes), self.status, self.rewires)
                    while len(self.attempts) > self.max_attempts:
                        self.attempts.popitem(last=False)
            self.dirty = True

    def accepted(self, epoch, result):
        with self.lock:
            if not self.enabled or epoch != self.epoch:
                return
            planned = result.get('result', {})
            if result.get('plan') and planned.get('playable') is True:
                selected = self.attempts.get(tuple(planned.get('goal_deg', [])))
                if selected:
                    self.nodes, self.status, self.rewires = copy.deepcopy(selected)
                # These frames have passed curve/encoder preparation and the
                # follower's current-session/fresh-target checks.
                self.route = [frame['q'] for frame in planned.get('frames', [])][:self.max_nodes]
                self.status = 'direct_checked' if self.status == 'direct_optimal' else 'checked_route'
                self.dirty = True

    def message(self):
        with self.lock:
            now = monotonic()
            if not self.dirty or (self.enabled and now-self.last_send < .1):
                return ''
            message = json.dumps({'schema_version':1, 'frame_id':'g_base',
                'enabled':self.enabled, 'epoch':self.epoch, 'attempt':self.attempt,
                'status':self.status, 'rewires':self.rewires,
                'nodes':self.nodes, 'route_deg':self.route}, allow_nan=False)
            self.last_send = now
            self.dirty = False
            return message
