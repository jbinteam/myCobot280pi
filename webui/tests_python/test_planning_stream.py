import json
import unittest
from webui.planning_stream import PlanningStream


class PlanningStreamTest(unittest.TestCase):
    def test_bounded_batches_preserve_events_and_end_with_checked_result(self):
        rows=[];stream=PlanningStream(rows.append)
        events=[{'kind':'begin'}]+[{'kind':'add','index':i} for i in range(70)]+[{'kind':'end'}]
        for event in events:stream.observe(event)
        stream.result({'playable':False,'status':'no_path'})
        messages=[json.loads(row) for row in rows]
        self.assertEqual([e for m in messages if m['type']=='rrt' for e in m['events']],events)
        self.assertTrue(all(len(m['events'])<=32 for m in messages if m['type']=='rrt'))
        self.assertEqual(messages[-1]['type'],'result')

    def test_disconnected_viewer_does_not_break_planning(self):
        def disconnected(_):raise BrokenPipeError()
        stream=PlanningStream(disconnected)
        stream.observe({'kind':'begin'})
        stream.observe({'kind':'add'})
        stream.result({'playable':True})
        self.assertTrue(stream.closed)
        self.assertEqual(stream.pending,[])
