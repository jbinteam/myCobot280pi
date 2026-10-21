"""Bounded six-joint RRT* with checked shortcuts and C2 segment timing."""
from dataclasses import dataclass, asdict
from time import perf_counter
import numpy as np
from ..kinematics import ARM


@dataclass(frozen=True)
class RRTConfig:
    iterations: int = 800
    step: float = .08
    radius: float = .2
    goal_bias: float = .18
    post_solution: int = 40
    neighbors: int = 20
    max_step_rad: float = .04

    def __post_init__(self):
        for key in ('iterations','post_solution','neighbors'):
            value=getattr(self,key)
            if isinstance(value,bool) or not isinstance(value,int) or value<(0 if key=='post_solution' else 1):
                raise ValueError(f'Invalid {key}')
        for key in ('step','radius','max_step_rad'):
            if not np.isfinite(getattr(self,key)) or getattr(self,key)<=0:raise ValueError(f'Invalid {key}')
        if not np.isfinite(self.goal_bias) or not 0<=self.goal_bias<=1:raise ValueError('Invalid goal bias')


class Tree:
    """RRT* vertices [N,6], parents and propagated root costs."""
    def __init__(self,start):
        self.q=[np.asarray(start,float).copy()];self.parent=[-1];self.cost=[0.];self.children=[set()]

    def add(self,q,parent,cost):
        index=len(self.q);self.q.append(q.copy());self.parent.append(parent);self.cost.append(float(cost))
        self.children.append(set());self.children[parent].add(index);return index

    def ancestor(self,node,possible):
        while node>=0:
            if node==possible:return True
            node=self.parent[node]
        return False

    def reparent(self,node,parent,cost):
        if node==0 or self.ancestor(parent,node):raise ValueError('Tree rewiring would create a cycle')
        delta=cost-self.cost[node]
        self.children[self.parent[node]].remove(node);self.parent[node]=parent;self.children[parent].add(node)
        pending=[node]
        while pending:
            index=pending.pop();self.cost[index]+=delta;pending.extend(self.children[index])

    def path(self,index):
        result=[]
        while index>=0:result.append(self.q[index]);index=self.parent[index]
        return np.asarray(result[::-1])


def plan(env,start,goal,seed=42,config=RRTConfig(),observer=None):
    """Return a checked path [T,6] and RRT* diagnostics; failure returns None."""
    began=perf_counter();start=np.asarray(start,float);goal=np.asarray(goal,float)
    report={'seed':seed,'configuration':asdict(config),'nodes':0,'iterations':0,'rewires':0}
    # Optional read-only diagnostics: never supply vertices to the planner.
    def emit(kind, **data):
        if observer is not None:observer({'kind':kind, **data})
    def finish(path, diagnostics):
        if observer is not None:
            emit('end',status=diagnostics['status'],nodes=diagnostics['nodes'],
                 iterations=diagnostics['iterations'],rewires=diagnostics['rewires'],
                 path_deg=None if path is None else np.rad2deg(path).tolist())
        return path, diagnostics
    if observer is not None:emit('begin',start_deg=np.rad2deg(start).tolist(),goal_deg=np.rad2deg(goal).tolist())
    if not env.valid(start) or not env.valid(goal):
        return finish(None,{**report,'status':'invalid_endpoint','seconds':perf_counter()-began})
    if env.edge_valid(start,goal,config.max_step_rad):
        # A straight segment attains the global lower bound in this metric.
        return finish(np.stack((start,goal)),{**report,'status':'direct_optimal','nodes':2,'seconds':perf_counter()-began})
    rng=np.random.default_rng(seed);tree=Tree(start);connections=[];first_solution=None
    scale=ARM.upper-ARM.lower
    metric=lambda a,b:np.linalg.norm((np.asarray(a)-b)/scale,axis=-1)
    for iteration in range(config.iterations):
        report['iterations']=iteration+1
        sample=goal if rng.random()<config.goal_bias else rng.uniform(ARM.lower,ARM.upper)
        values=np.asarray(tree.q);distance=metric(values,sample);nearest=int(np.argmin(distance))
        delta=sample-values[nearest]
        q=values[nearest]+delta*min(1.,config.step/max(float(distance[nearest]),1e-12))
        if distance[nearest]<1e-8 or not env.edge_valid(values[nearest],q,config.max_step_rad):continue
        distances=metric(values,q);n=len(values)+1
        radius=min(config.radius,.5*(np.log(n+1)/n)**(1/6))
        nearby=np.flatnonzero(distances<=radius)
        nearby=nearby[np.argsort(distances[nearby])[:config.neighbors]]
        costs=np.asarray(tree.cost)
        choices=sorted(set(nearby.tolist()+[nearest]),key=lambda i:costs[i]+distances[i])
        parent=nearest;cost=costs[nearest]+distances[nearest]
        for index in choices:
            if index==nearest or env.edge_valid(values[index],q,config.max_step_rad):
                parent=index;cost=costs[index]+distances[index];break
        node=tree.add(q,parent,cost)
        if observer is not None:emit('add',index=node,parent=parent,q_deg=np.rad2deg(q).tolist())
        for index in nearby:
            new_cost=tree.cost[node]+distances[index]
            if new_cost<tree.cost[index]-1e-10 and not tree.ancestor(node,int(index)) and env.edge_valid(q,values[index],config.max_step_rad):
                tree.reparent(int(index),node,float(new_cost));report['rewires']+=1
                if observer is not None:emit('rewire',index=int(index),parent=node)
        if metric(q,goal)<=config.radius or iteration%10==0:
            if env.edge_valid(q,goal,config.max_step_rad):
                connections.append(node)
                if first_solution is None:first_solution=iteration
        if first_solution is not None and iteration-first_solution>=config.post_solution:break
    report.update(nodes=len(tree.q),seconds=perf_counter()-began)
    if not connections:return finish(None,{**report,'status':'iteration_limit'})
    winner=min(connections,key=lambda i:tree.cost[i]+metric(tree.q[i],goal))
    path=np.vstack((tree.path(winner),goal))
    if not all(env.edge_valid(a,b,config.max_step_rad) for a,b in zip(path[:-1],path[1:])):
        raise RuntimeError('RRT* final path failed collision revalidation')
    return finish(path,{**report,'status':'rrt_star','cost':float(tree.cost[winner]+metric(tree.q[winner],goal)),
                 'first_solution_iteration':first_solution+1})


def smooth(env,path,max_step_rad=.04):
    """Greedy shortcuts, each newly checked; retain original safe corners."""
    path=np.asarray(path,float);result=[path[0]];index=0
    while index<len(path)-1:
        next_index=index+1
        for target in range(len(path)-1,index,-1):
            if env.edge_valid(path[index],path[target],max_step_rad):next_index=target;break
        result.append(path[next_index]);index=next_index
    return np.asarray(result)


def trajectory(path,dt=.1,speed=1.,acceleration=3.):
    """C2 monotone quintics along checked straight segments; return [T,6], [T].

    Each segment stops at its ends. Analytic derivative bounds constrain the
    six-joint vector speed and acceleration. No spline may cut a PRM/RRT corner.
    """
    path=np.asarray(path,float)
    if path.ndim!=2 or path.shape[1]!=6 or len(path)<2 or not np.isfinite(path).all():raise ValueError('Expected finite [T,6] path')
    if not all(np.isfinite(x) and x>0 for x in (dt,speed,acceleration)):raise ValueError('Positive timing limits required')
    if np.any(path<ARM.lower) or np.any(path>ARM.upper):raise ValueError('Path violates operating limits')
    rows=[path[0]];times=[0.]
    for a,b in zip(path[:-1],path[1:]):
        distance=float(np.linalg.norm(b-a))
        if distance<1e-12:continue
        duration=max(1.875*distance/speed,np.sqrt(5.773503*distance/acceleration))*1.001
        pieces=max(1,int(np.ceil(duration/dt)));duration=pieces*dt
        u=np.arange(1,pieces+1)/pieces;fraction=10*u**3-15*u**4+6*u**5
        segment=a+fraction[:,None]*(b-a);segment[-1]=b
        rows.extend(segment);times.extend(times[-1]+np.arange(1,pieces+1)*dt)
    if len(rows)==1:rows.append(path[-1]);times.append(dt)
    return np.asarray(rows),np.asarray(times)
