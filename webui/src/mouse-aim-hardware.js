// Server-owned sessions and plans; this module never sends joint arrays.
export class HardwareMouseAim {
  constructor({ getState, onState, onFault, onComplete }) {
    Object.assign(this, { getState, onState, onFault, onComplete });
    this.sessionId = null; this.executionId = null; this.pendingExecute = false;
    this.generation = 0; this.startedUpdated = 0; this.heartbeatBusy = false;
    setInterval(() => this.heartbeat(), 1000);
  }
  get running() { return this.pendingExecute || !!this.executionId; }
  async request(path, body, signal) {
    const response = await fetch(`/api/ik/${path}`, { method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Robot-Token': this.getState().token },
      body: JSON.stringify(body), signal: signal || AbortSignal.timeout(path==='mouse-aim/plan'?90000:12000) });
    const data = await response.json();
    if (!response.ok) throw Object.assign(new Error(data.error || 'Robot mouse aim request failed'), { status: response.status });
    return data;
  }
  async start(settings={}) {
    const epoch = ++this.generation;
    const data = await this.request('mouse-aim/start', { armed: true, frame_verified: true, ...settings });
    if (epoch !== this.generation) {
      await this.request('mouse-aim/stop', { session_id: data.session_id });
      return null;
    }
    if (!data.session_id || !Array.isArray(data.angles)) throw new Error('Robot did not return a live mouse aim session');
    if(['diffusion_v2','learned_rrt'].includes(settings.tracking_engine)&&data.ik?.mouse_encoder_profile_protocol!==1){
      await this.request('mouse-aim/stop',{session_id:data.session_id});
      throw new Error('Restart the control bridge to enable the shared encoder velocity executor');
    }
    if(['diffusion_v2','learned_rrt'].includes(settings.tracking_engine)
      &&(data.ik?.mouse_feedback_tracking_protocol!==1||data.ik?.mouse_executor!=='encoder_feedback_tracking')){
      await this.request('mouse-aim/stop',{session_id:data.session_id});
      throw new Error('Restart the control bridge to enable feedback-paced gaze tracking');
    }
    if(settings.tracking_region&&(data.ik?.mouse_tracking_region_protocol!==1||JSON.stringify(data.ik?.tracking_region_settings)!==JSON.stringify(settings.tracking_region))){
      await this.request('mouse-aim/stop',{session_id:data.session_id});
      throw new Error('Restart the control bridge to enable cone tracking regions');
    }
    if(settings.velocity_shape&&(data.ik?.mouse_velocity_profile_protocol!==1||JSON.stringify(data.ik?.velocity_shape)!==JSON.stringify(settings.velocity_shape))){
      await this.request('mouse-aim/stop', {session_id:data.session_id});
      throw new Error('Restart the control bridge to enable the shared mouse velocity profile');
    }
    if (data.ik?.tracking_protocol !== 2 || (['diffusion_v2','learned_rrt'].includes(settings.tracking_engine) && (data.ik?.v2_tracking_protocol!==1||data.ik?.mouse_engine!==settings.tracking_engine))) {
      await this.request('mouse-aim/stop', { session_id: data.session_id });
      throw new Error('Restart the control bridge to enable verified velocity tracking');
    }
    this.sessionId = data.session_id; this.startedUpdated = data.updated;
    this.onState(data);
    return data;
  }
  async plan(cursor, settings, signal) {
    if (!this.sessionId) throw new Error('Start robot mouse aim again');
    const id = this.sessionId;
    const data = await this.request('mouse-aim/plan', { session_id: id, cursor_mm: cursor, ...settings }, signal);
    if (data.session_id !== id || this.sessionId !== id) throw new Error('Mouse aim session changed while planning');
    if (data.hardware_mouse_aim !== true || data.simulation_only !== false || !Array.isArray(data.start_deg) || data.start_deg.length !== 6) {
      throw new Error('Robot did not return a live-start hardware mouse aim plan');
    }
    if(['learned_rrt','diffusion_v2'].includes(this.getState().ik?.mouse_engine)&&data.plan
      &&(data.plan.execution_mode!=='encoder_feedback_tracking'||data.plan.joint_command!=='encoder_drag_profile'))throw new Error('Tracking plan did not use feedback-paced gaze tracking');
    return data;
  }
  async execute(plan, speed) {
    if (!this.sessionId || !plan?.id || !(plan.converged||(plan.hardware_v2_mouse&&plan.playable))) throw new Error('No accepted server-owned mouse aim plan');
    if (!Number.isInteger(speed) || speed < 1 || speed > 80) throw new Error('IK SDK speed must be an integer from 1 to 80');
    const epoch = this.generation, id = this.sessionId;
    this.pendingExecute = true;
    try {
      const data = await this.request('execute', { plan_id: plan.id, session_id:id, speed, armed: true, frame_verified: true,
        ...(plan.hardware_v2_mouse?{scene_verified:true}:{}) });
      if (epoch !== this.generation) return;
      if ((data.ik?.execution_id !== plan.id && data.ik?.accepted_plan_id !== plan.id) || data.ik?.mouse_aim_session_id !== id) throw new Error('Robot returned a different execution');
      // Only a received execute response permits the first motor heartbeat.
      this.executionId = data.ik.execution_id;
      this.onState(data);
      // Acknowledge immediately after the received response, without the
      // otherwise up-to-one-second wait for the periodic lease heartbeat.
      if (epoch === this.generation) {
        try { await this.request('mouse-aim/heartbeat', { session_id: id, execution_id: this.executionId,
          ...(this.executionId !== plan.id ? { plan_id: plan.id } : {}) }); }
        catch (error) {
          if (epoch !== this.generation) return;
          // A concurrent periodic acknowledgement can finish an empty/short
          // path before this acknowledgement reaches the server.
          const next = await (await fetch('/api/state', { signal: AbortSignal.timeout(3000) })).json();
          if (epoch !== this.generation) return;
          if (next.ik?.execution_id !== this.executionId || next.ik?.mouse_aim_session_id !== id || next.ik?.state !== 'completed'
              || (this.executionId !== plan.id && next.ik?.active_plan_id !== plan.id)) throw error;
          this.onState(next); this.observe(next);
        }
      }
      if (epoch !== this.generation) return;
      window.dispatchEvent(new CustomEvent('robot-state-refresh'));
    } finally { if (epoch === this.generation) this.pendingExecute = false; }
  }
  observe(next) {
    if (!this.sessionId || next.updated < this.startedUpdated) return;
    if (!next.connected || next.ik?.mouse_aim_session_id !== this.sessionId) {
      this.onFault(next.error || next.ik?.message || 'Robot mouse aim session ended'); return;
    }
    if (this.executionId && next.ik?.execution_id === this.executionId) {
      if (next.ik.state === 'completed') {
        this.executionId = null; this.onComplete(next.ik);
      } else if (['failed', 'stopped', 'disconnected'].includes(next.ik.state)) {
        this.onFault(next.ik.message || 'Robot mouse aim stopped');
      }
    }
  }
  async heartbeat() {
    if (!this.sessionId || this.heartbeatBusy) return;
    this.heartbeatBusy = true;
    const epoch = this.generation;
    const controller = new AbortController(); this.heartbeatController = controller;
    try {
      await this.request('mouse-aim/heartbeat', { session_id: this.sessionId,
        ...(this.executionId ? { execution_id: this.executionId } : {}) },
        AbortSignal.any([controller.signal, AbortSignal.timeout(12000)]));
    } catch (error) {
      if (epoch !== this.generation) return;
      try {
        const next = await (await fetch('/api/state', { signal: AbortSignal.timeout(3000) })).json();
        if (epoch !== this.generation) return;
        this.observe(next);
        if (this.sessionId && !this.executionId && next.ik?.state === 'completed') return;
      } catch { /* The server lease stops motion when the bridge is unreachable. */ }
      if (epoch === this.generation) this.onFault(`Robot mouse aim heartbeat failed: ${error.message}`);
    } finally { if (epoch === this.generation) { this.heartbeatBusy = false; this.heartbeatController = null; } }
  }
  async stop() {
    const epoch = ++this.generation, id = this.sessionId;
    this.heartbeatController?.abort(); this.heartbeatController = null; this.heartbeatBusy = false;
    this.sessionId = this.executionId = null; this.pendingExecute = false;
    if (!id) return;
    try {
      const next = await this.request('mouse-aim/stop', { session_id: id });
      if (epoch === this.generation) this.onState(next);
    } catch { /* A missing acknowledgement leaves no heartbeat; the server stops. */ }
  }
}
