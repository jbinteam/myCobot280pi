import json
from object_tracking_ros2.planning_visualization import PlanningVisualization


def begin(observer, goal=1):
    observer({'kind':'begin', 'start_deg':[0.]*6, 'goal_deg':[float(goal)]*6})


def test_disabled_collection_stale_callbacks_and_current_checked_selection():
    view = PlanningVisualization()
    assert view.begin() is None
    view.set_enabled(True)
    epoch, observe = view.begin()
    begin(observe)
    observe({'kind':'add', 'index':1, 'parent':0, 'q_deg':[.5]*6})
    observe({'kind':'end', 'status':'success', 'rewires':2})
    begin(observe, 2)
    observe({'kind':'end', 'status':'no_path', 'rewires':0})
    view.accepted(epoch, {'plan':{'id':'checked'}, 'result':{'playable':True,
        'goal_deg':[1.]*6, 'frames':[{'q':[0.]*6}, {'q':[1.]*6}]}})
    snapshot = json.loads(view.message())
    assert snapshot['nodes'][1]['q_deg'] == [.5]*6
    assert snapshot['rewires'] == 2 and len(snapshot['route_deg']) == 2
    assert not view.message()  # dirty/coalescing gate
    view.set_enabled(False)
    observe({'kind':'add', 'index':2, 'parent':1, 'q_deg':[1.]*6})
    view.accepted(epoch, {'plan':{}, 'result':{'playable':True}})
    snapshot = json.loads(view.message())
    assert not snapshot['enabled'] and not snapshot['nodes'] and not snapshot['route_deg']
    view.set_enabled(True)
    old_epoch = epoch
    epoch, current = view.begin()
    begin(current, 3)
    view.observe(old_epoch, {'kind':'begin', 'start_deg':[9.]*6, 'goal_deg':[9.]*6})
    assert view.nodes[0]['q_deg'] == [0.]*6


def test_bounds_direct_candidates_and_unplayable_results_have_no_route():
    view = PlanningVisualization()
    view.max_nodes = 3
    view.max_attempts = 2
    view.set_enabled(True)
    epoch, observe = view.begin()
    for goal in range(4):
        begin(observe, goal)
        for index in range(1, 8):
            observe({'kind':'add', 'index':index, 'parent':0, 'q_deg':[float(index)]*6})
        observe({'kind':'end', 'status':'no_path', 'rewires':0})
    assert len(view.nodes) == 3 and len(view.attempts) == 2
    view.accepted(epoch, {'result':{'playable':False, 'frames':[{'q':[0.]*6}]}})
    assert not view.route
    begin(observe, 5)
    observe({'kind':'end', 'status':'direct_optimal', 'rewires':0,
             'path_deg':[[0.]*6, [5.]*6]})
    view.accepted(epoch, {'plan':{'id':'checked'}, 'result':{'playable':True,
        'goal_deg':[5.]*6, 'frames':[{'q':[0.]*6}, {'q':[5.]*6}]}})
    assert view.status == 'direct_checked' and len(view.nodes) == 2
