const $ = id => document.getElementById(id);

export function initRobotControls({ getPose, setPose, setEditing }) {
  let state = { connected: false }, token = '', polling = false, busy = false, limits = null;
  let latestAngles = null, lastUpdated = 0;
  let mouseTracking = false;
  let switchingMode = false, modeError = '';
  const movementButtons = ['send-pose', 'jog-negative', 'jog-positive'];
  const feedback = (message, error = false) => {
    $('robot-feedback').textContent = message;
    $('robot-feedback').classList.toggle('error', error);
  };
  function updateButtons() {
    const connected = state.connected;
    const enabled = state.hardware_enabled !== false;
    const modelReady = $('model-status').textContent === 'Model ready';
    const running = state.moving || state.motion;
    $('control-mode').disabled = !token || busy || switchingMode || typeof state.hardware_enabled !== 'boolean';
    $('control-mode').querySelector('[value="hardware"]').disabled = state.mode_selectable === false;
    $('connect-robot').disabled = !enabled || connected || busy || !token || !modelReady;
    $('disconnect-robot').disabled = !connected || busy;
    $('robot-host').disabled = connected || busy;
    $('robot-port').disabled = connected || busy;
    $('stop-robot').disabled = !connected;
    $('arm-movement').disabled = !connected;
    $('use-live-pose').disabled = !connected || !latestAngles;
    $('follow-live').disabled = !connected || mouseTracking;
    for (const id of movementButtons) $(id).disabled = !connected || !$('arm-movement').checked || !!running || busy || mouseTracking || !modelReady;
    for (const id of ['jog-negative', 'jog-positive']) {
      if (state.directional_jog_blocked) $(id).disabled = true;
      $(id).title = state.directional_jog_blocked
        ? 'Directional jog disabled on Atom 6.5 after excessive travel; firmware validation and a Pi-side Stop watchdog are required.' : '';
    }
  }
  function syncViewMode() {
    const follow = state.connected && $('follow-live').checked;
    setEditing(!follow);
    $('mode-badge').textContent = state.hardware_enabled === false ? '◈ Simulation only · hardware disabled' : follow ? '◈ Live robot pose' : '◈ Preview mode';
    $('mode-badge').classList.toggle('live-mode', follow);
    if (follow && latestAngles) setPose(latestAngles, true);
  }
  function applyState(next) {
    const wasConnected = state.connected;
    const changedMode = typeof state.hardware_enabled === 'boolean' && typeof next.hardware_enabled === 'boolean'
      && state.hardware_enabled !== next.hardware_enabled;
    state = next;
    if (changedMode && !switchingMode) {
      window.dispatchEvent(new Event('control-mode-changing'));
      location.reload();
      return;
    }
    if (!switchingMode) $('control-mode').value = next.hardware_enabled === true ? 'hardware' : 'simulation';
    if (!switchingMode) $('control-mode-note').textContent = modeError || (next.mode_selectable === false
      ? 'Locked to simulation. Start ./start_v2.sh for selectable modes.'
      : next.hardware_enabled === true ? 'Connect and confirm the setup before moving the real arm.'
      : 'Preview only. Robot connection and movement commands are disabled.');
    document.querySelector('.robot-toolbar').hidden = next.hardware_enabled === false;
    document.querySelector('.robot-actions').hidden = next.hardware_enabled === false;
    if (next.token) token = next.token;
    if (next.limits) limits = next.limits;
    if (next.connected && Array.isArray(next.angles) && next.angles.length === 6) {
      latestAngles = next.angles;
      lastUpdated = next.updated * 1000;
      $('live-angles').textContent = JSON.stringify(latestAngles.map(angle => Number(angle.toFixed(1))));
      $('connection-status').textContent = `Connected · Atom ${next.firmware ?? '—'}`;
      $('connection-dot').classList.add('ready');
      $('robot-activity').textContent = next.motion === 'jog' ? 'Jogging' : (next.motion || next.moving) ? 'Moving' : 'Stopped';
      if (!wasConnected) {
        $('arm-movement').checked = false;
        $('follow-live').checked = true;
      }
    } else {
      latestAngles = null;
      $('connection-status').textContent = 'Robot disconnected';
      $('connection-dot').classList.remove('ready');
      $('live-angles').textContent = 'Not connected';
      $('robot-activity').textContent = '—';
      $('arm-movement').checked = false;
      $('follow-live').checked = false;
    }
    if (next.error) feedback(next.error, true);
    syncViewMode();
    updateButtons();
    window.dispatchEvent(new CustomEvent('robot-state', { detail: next }));
  }
  async function request(path, body) {
    const response = await fetch(`/api/${path}`, {
      method: body ? 'POST' : 'GET',
      headers: body ? { 'Content-Type': 'application/json', 'X-Robot-Token': token } : {},
      body: body ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(12000),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `Robot request failed (${response.status})`);
    return data;
  }
  async function command(path, body, message) {
    if (state.hardware_enabled === false) return;
    busy = true;
    updateButtons();
    try {
      const next = await request(path, body);
      applyState(next);
      if (!next.error) feedback(message);
    } catch (error) { feedback(error.message, true); }
    finally { busy = false; updateButtons(); }
  }
  function motionSettings() {
    if (!state.connected) throw new Error('Connect to the robot first');
    if (!lastUpdated || Date.now() - lastUpdated > 3000) throw new Error('Live readings are stale. Reconnect before moving.');
    if (!$('arm-movement').checked) throw new Error('Enable movement after clearing the arm path');
    const speed = Number($('robot-speed').value);
    if (!Number.isInteger(speed) || speed < 1 || speed > 80) throw new Error('Speed must be an integer from 1 to 80');
    return { speed, armed: true };
  }
  $('control-mode').addEventListener('change', async () => {
    const mode = $('control-mode').value;
    modeError = '';
    switchingMode = true; busy = true; updateButtons();
    $('control-mode-note').textContent = mode === 'simulation'
      ? 'Stopping tracking and disconnecting the arm…' : 'Switching to Simulation + hardware…';
    window.dispatchEvent(new Event('control-mode-changing'));
    try {
      await request('mode', { mode });
      // Start with fresh controllers, no pending previews, plans or checked
      // movement permissions. The backend is authoritative across browser tabs.
      location.reload();
    } catch (error) {
      switchingMode = false; busy = false;
      modeError = error.message;
      $('control-mode').value = state.hardware_enabled === true ? 'hardware' : 'simulation';
      $('control-mode-note').textContent = error.message;
      updateButtons();
    }
  });
  $('connect-robot').addEventListener('click', () => command('connect', {
    host: $('robot-host').value.trim(), port: Number($('robot-port').value),
  }, 'Connected. The model follows live readings. Uncheck Follow robot to edit a target pose.'));
  $('disconnect-robot').addEventListener('click', () => command('disconnect', {}, 'Disconnected; any movement started here was stopped.'));
  $('stop-robot').addEventListener('click', () => command('stop', {}, 'Stop command sent; check the live motion status.'));
  $('arm-movement').addEventListener('change', () => {
    updateButtons();
    if (!$('arm-movement').checked && state.connected && (state.motion || state.moving)) {
      command('stop', {}, 'Movement disabled; stop command sent.');
    }
  });
  $('follow-live').addEventListener('change', syncViewMode);
  $('use-live-pose').addEventListener('click', () => { if (latestAngles) setPose(latestAngles, true); });
  $('send-pose').addEventListener('click', () => {
    try {
      const settings = motionSettings(), angles = getPose();
      if (!limits || angles.some((a, i) => !Number.isFinite(a) || a < limits.min[i] || a > limits.max[i])) {
        throw new Error('Preview exceeds the robot SDK joint limits. Adjust the angles before moving.');
      }
      command('move', { ...settings, angles }, `Move sent at speed ${settings.speed}. Live readings show the actual pose.`);
    } catch (error) { feedback(error.message, true); }
  });
  for (const [id, direction] of [['jog-negative', 0], ['jog-positive', 1]]) {
    $(id).addEventListener('click', () => {
      try {
        const settings = motionSettings(), duration = Number($('jog-duration').value);
        if (!Number.isFinite(duration) || duration < 0.1 || duration > 2) throw new Error('Jog duration must be 0.1–2 seconds');
        command('jog', { ...settings, joint: Number($('jog-joint').value), direction, duration }, `Jog sent for ${duration}s; the server will stop it automatically.`);
      } catch (error) { feedback(error.message, true); }
    });
  }
  async function poll() {
    if (polling || busy) return;
    polling = true;
    try { applyState(await request('state')); }
    catch {
      token = '';
      applyState({ connected: false });
      feedback('Control bridge unavailable. Start: .venv/bin/python webui/server.py', true);
    } finally { polling = false; }
  }
  window.addEventListener('robot-state-refresh', poll);
  window.addEventListener('mouse-aim-change', event => { mouseTracking = event.detail.active && event.detail.hardware; updateButtons(); });
  poll();
  setInterval(poll, 750);
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && state.connected) $('stop-robot').click();
  });
}
