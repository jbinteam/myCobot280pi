import { test, expect } from '@playwright/test';

test('connects read-only, requires movement enable, and sends explicit commands', async ({ page }, testInfo) => {
  const calls = [];
  let state = { hardware_enabled: true, connected: false, token: 'test-session', host: '192.168.77.225', port: 9000, angles: null, moving: false, motion: null, error: null, firmware: 6.5, limits: { min: [-168,-140,-150,-150,-155,-180], max: [168,140,150,150,160,180] } };
  await page.route('**/api/**', async route => {
    const request = route.request(), path = new URL(request.url()).pathname;
    if (request.method() === 'POST') {
      const body = request.postDataJSON(); calls.push({path,body});
      expect(request.headers()['x-robot-token']).toBe('test-session');
      if (path === '/api/connect') state = { ...state, connected: true, angles: [1,2,3,4,5,6] };
      if (path === '/api/move') state = { ...state, motion: 'pose', moving: true };
      if (path === '/api/jog') state = { ...state, motion: 'jog', moving: true };
      if (path === '/api/stop') state = { ...state, motion: null, moving: false };
      if (path === '/api/disconnect') state = { ...state, connected: false };
    }
    await route.fulfill({ json: { ...state, updated: Date.now()/1000 } });
  });
  await page.goto('/?view=lab');
  await expect(page.locator('#model-status')).toHaveText('Model ready', { timeout: 45000 });
  await expect(page.locator('#connect-robot')).toBeEnabled();
  expect(calls).toEqual([]);
  await page.locator('#connect-robot').click();
  await expect(page.locator('#connection-status')).toContainText('Connected');
  await expect(page.locator('#live-angles')).toHaveText('[1,2,3,4,5,6]');
  await expect(page.locator('#pose-value')).toHaveText('[1,2,3,4,5,6]');
  await expect(page.locator('#send-pose')).toBeDisabled();
  await expect(page.locator('#joint-0-slider')).toBeDisabled();
  await page.locator('#follow-live').uncheck();
  await page.locator('#rest-pose').click();
  await expect(page.locator('#pose-value')).toHaveText('[0,-40,-130,80,0,50]');
  expect(calls.map(call=>call.path)).toEqual(['/api/connect']);
  await page.locator('#arm-movement').check();
  await page.locator('#send-pose').click();
  await expect(page.locator('#robot-activity')).toHaveText('Moving');
  expect(calls[1]).toEqual({path:'/api/move',body:{angles:[0,-40,-130,80,0,50],speed:30,armed:true}});
  await expect(page.locator('#send-pose')).toBeDisabled();
  await expect(page.locator('#stop-robot')).toBeEnabled();
  await page.screenshot({ path: testInfo.outputPath('control-ui.png'), fullPage: true });
  await page.locator('#stop-robot').click();
  await expect(page.locator('#robot-activity')).toHaveText('Stopped');
  await page.locator('#jog-joint').selectOption('3');
  await page.locator('#jog-negative').click();
  await expect(page.locator('#robot-activity')).toHaveText('Jogging');
  expect(calls.at(-1)).toEqual({path:'/api/jog',body:{joint:3,direction:0,speed:30,duration:0.5,armed:true}});
  await page.keyboard.press('Escape');
  await expect(page.locator('#robot-activity')).toHaveText('Stopped');
  await page.locator('#jog-positive').click();
  await expect(page.locator('#robot-activity')).toHaveText('Jogging');
  await page.locator('#arm-movement').uncheck();
  await expect(page.locator('#robot-activity')).toHaveText('Stopped');
  expect(calls.at(-1).path).toBe('/api/stop');
  await page.locator('#disconnect-robot').click();
  await expect(page.locator('#arm-movement')).not.toBeChecked();
  await expect(page.locator('#send-pose')).toBeDisabled();
});

test('connection errors stay visible and motion remains disabled', async ({ page }) => {
  await page.route('**/api/**', async route => {
    if (route.request().method() === 'GET') await route.fulfill({json:{connected:false,token:'test-session'}});
    else await route.fulfill({status:503,json:{error:'Connection refused'}});
  });
  await page.goto('/?view=lab');
  await expect(page.locator('#connect-robot')).toBeEnabled({timeout:45000});
  await page.locator('#connect-robot').click();
  await expect(page.locator('#robot-feedback')).toHaveText('Connection refused');
  await expect(page.locator('#send-pose')).toBeDisabled();
  await expect(page.locator('#arm-movement')).toBeDisabled();
});
