import { test, expect } from '@playwright/test';

test('received hardware execution is acknowledged without waiting for the periodic timer', async ({ page }) => {
  await page.goto('/?view=lab');
  const result = await page.evaluate(async () => {
    const { HardwareMouseAim } = await import('/src/mouse-aim-hardware.js');
    const realFetch = window.fetch, realInterval = window.setInterval;
    const posts = [], callbacks = [];
    window.setInterval = callback => { callbacks.push(callback); return 0; };
    window.fetch = async (url, options) => {
      posts.push({ url, body: JSON.parse(options.body) });
      return { ok: true, json: async () => url.endsWith('/execute')
        ? { ik: { execution_id: 'test-plan', mouse_aim_session_id: 'test-session' } } : { active: true } };
    };
    try {
      const aim = new HardwareMouseAim({ getState: () => ({ token: 'test-token' }), onState: () => {}, onFault: () => {}, onComplete: () => {} });
      aim.sessionId = 'test-session';
      await aim.execute({ id: 'test-plan', converged: true }, 10);
      return { posts, periodicCallbacks: callbacks.length };
    } finally { window.fetch = realFetch; window.setInterval = realInterval; }
  });
  expect(result.periodicCallbacks).toBe(1);
  expect(result.posts).toEqual([
    { url: '/api/ik/execute', body: { plan_id: 'test-plan', speed: 10, armed: true, frame_verified: true } },
    { url: '/api/ik/mouse-aim/heartbeat', body: { session_id: 'test-session', execution_id: 'test-plan' } },
  ]);
});

test('immediate acknowledgement accepts an already completed owned path', async ({ page }) => {
  await page.goto('/?view=lab');
  const result = await page.evaluate(async () => {
    const { HardwareMouseAim } = await import('/src/mouse-aim-hardware.js');
    const realFetch = window.fetch, realInterval = window.setInterval;
    const posts = []; let completed = 0;
    const state = { connected: true, updated: Date.now()/1000,
      ik: { state: 'completed', execution_id: 'test-plan', mouse_aim_session_id: 'test-session' } };
    window.setInterval = () => 0;
    window.fetch = async (url, options) => {
      posts.push(url);
      return { ok: !url.endsWith('/heartbeat'), json: async () => url.endsWith('/execute')
        ? { ...state, ik: { ...state.ik, state: 'running' } } : url === '/api/state' ? state : { error: 'Execution is no longer active' } };
    };
    try {
      const aim = new HardwareMouseAim({ getState: () => ({ token: 'test-token' }), onState: () => {}, onFault: () => {}, onComplete: () => completed++ });
      aim.sessionId = 'test-session';
      await aim.execute({ id: 'test-plan', converged: true }, 10);
      return { posts, completed, running: aim.running };
    } finally { window.fetch = realFetch; window.setInterval = realInterval; }
  });
  expect(result.posts).toEqual(['/api/ik/execute', '/api/ik/mouse-aim/heartbeat', '/api/state']);
  expect(result.completed).toBe(1);
  expect(result.running).toBe(false);
});

test('a late replacement response after Stop cannot acknowledge a new route', async ({ page }) => {
  await page.goto('/?view=lab');
  const result = await page.evaluate(async () => {
    const { HardwareMouseAim } = await import('/src/mouse-aim-hardware.js');
    const realFetch = window.fetch, realInterval = window.setInterval;
    const posts = []; let release;
    const held = new Promise(resolve => { release = resolve; });
    window.setInterval = () => 0;
    window.fetch = async (url, options) => {
      posts.push({ url, body: JSON.parse(options.body) });
      if (url.endsWith('/execute')) await held;
      return { ok: true, json: async () => url.endsWith('/execute')
        ? { ik: { execution_id: 'root', accepted_plan_id: 'replacement', mouse_aim_session_id: 'session' } } : { active: true } };
    };
    try {
      const aim = new HardwareMouseAim({ getState: () => ({ token: 'test' }), onState: () => {}, onFault: () => {}, onComplete: () => {} });
      aim.sessionId = 'session'; aim.executionId = 'root';
      const pending = aim.execute({ id: 'replacement', converged: true }, 10);
      await aim.heartbeat();
      await aim.stop();
      release(); await pending;
      return { posts, running: aim.running };
    } finally { window.fetch = realFetch; window.setInterval = realInterval; }
  });
  expect(result.posts).toEqual([
    { url: '/api/ik/execute', body: { plan_id: 'replacement', speed: 10, armed: true, frame_verified: true } },
    { url: '/api/ik/mouse-aim/heartbeat', body: { session_id: 'session', execution_id: 'root' } },
    { url: '/api/ik/mouse-aim/stop', body: { session_id: 'session' } },
  ]);
  expect(result.running).toBe(false);
});

test('old bridge protocols cannot start velocity tracking', async ({ page }) => {
  await page.goto('/?view=lab');
  const result = await page.evaluate(async () => {
    const { HardwareMouseAim } = await import('/src/mouse-aim-hardware.js');
    const realFetch = window.fetch, realInterval = window.setInterval, posts = [];
    window.setInterval = () => 0;
    window.fetch = async (url, options) => {
      posts.push(url);
      return { ok: true, json: async () => url.endsWith('/start')
        ? { session_id: 'old-session', angles: [0,0,0,0,0,0], ik: { tracking_protocol: 1 } } : {} };
    };
    try {
      const aim = new HardwareMouseAim({ getState: () => ({ token: 'test' }), onState: () => {}, onFault: () => {}, onComplete: () => {} });
      let error; try { await aim.start(); } catch (e) { error = e.message; }
      return { posts, error, session: aim.sessionId };
    } finally { window.fetch = realFetch; window.setInterval = realInterval; }
  });
  expect(result.posts).toEqual(['/api/ik/mouse-aim/start', '/api/ik/mouse-aim/stop']);
  expect(result.error).toContain('Restart the control bridge');
  expect(result.session).toBeNull();
});
