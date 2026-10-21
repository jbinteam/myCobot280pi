import { test, expect } from '@playwright/test';
import { execFileSync } from 'node:child_process';

let info;
test.beforeAll(() => {
  info = JSON.parse(execFileSync('.venv/bin/python', ['-c',
    'import json; from simulation.service import info; print(json.dumps(info()))'],
    { cwd:'..', encoding:'utf8', env:{...process.env, OPENBLAS_NUM_THREADS:'1'} }));
});

async function setup(page, { locked=false, fail=false }={}) {
  const writes=[], errors=[];
  const state={ hardware_enabled:false, mode_selectable:!locked, control_mode:'simulation',
    connected:false, token:'fake-mode-token', motion:null, moving:false, angles:null,
    firmware:6.5, limits:info.limits_deg, ik:{ state:'idle', samples:[], plan:null,
      tracking_protocol:2, v2_tracking_protocol:1 } };
  page.on('pageerror', e => errors.push(e.message));
  await page.route('**/api/**', async route => {
    const request=route.request(), path=new URL(request.url()).pathname;
    if (path==='/api/sim/info') return route.fulfill({json:info});
    if (request.method()==='POST') {
      const body=request.postDataJSON(); writes.push({path,body});
      expect(request.headers()['x-robot-token']).toBe(state.token);
      if (path==='/api/mode') {
        if (fail) return route.fulfill({status:503,json:{error:'Mode change failed'}});
        state.hardware_enabled=body.mode==='hardware'; state.control_mode=body.mode;
        state.connected=false; state.angles=null; state.motion=null;
      } else if (path==='/api/connect' && state.hardware_enabled) {
        state.connected=true; state.angles=info.default_start_deg;
      } else throw new Error(`Unexpected write: ${path}`);
    } else if (path!=='/api/state') throw new Error(`Unexpected read: ${path}`);
    return route.fulfill({json:{...state,updated:Date.now()/1000}});
  });
  await page.goto('/');
  await expect(page.locator('#recommended-model-note')).toContainText('Best qualified');
  await expect(page.locator('#control-mode')).toBeEnabled();
  return {writes,errors,state};
}

async function ready(page, mode) {
  await expect(page.locator('#control-mode')).toHaveValue(mode);
  await expect(page.locator('#recommended-model-note')).toContainText('Best qualified');
  await expect(page.locator('#control-mode')).toBeEnabled();
}

test('switches both ways without connecting automatically and clears all movement confirmations', async ({page},testInfo) => {
  const app=await setup(page);
  await page.locator('#toggle-mouse-aim').click();
  await expect(page.locator('#toggle-mouse-aim')).toHaveAttribute('aria-pressed','true');
  await page.locator('#control-mode').selectOption('hardware');
  await ready(page,'hardware');
  await expect(page.locator('.robot-toolbar')).toBeVisible();
  await expect(page.locator('#hardware-ik')).toBeVisible();
  await expect(page.locator('#simulation-setup')).toBeHidden();
  await expect(page.locator('#toggle-mouse-aim')).toHaveAttribute('aria-pressed','false');
  await expect(page.locator('#toggle-mouse-aim')).toBeDisabled();
  expect(app.writes).toEqual([{path:'/api/mode',body:{mode:'hardware'}}]);
  await page.locator('#connect-robot').click();
  await page.locator('#verify-ik-frame').check();
  await page.locator('#verify-diffusion-scene').check();
  await page.locator('#arm-movement').check();
  await expect(page.locator('#toggle-mouse-aim')).toBeEnabled();
  await page.locator('#control-mode').selectOption('simulation');
  await ready(page,'simulation');
  await expect(page.locator('.robot-toolbar')).toBeHidden();
  await expect(page.locator('#hardware-ik')).toBeHidden();
  await expect(page.locator('#simulation-setup')).toBeVisible();
  for (const id of ['verify-ik-frame','verify-diffusion-scene','arm-movement'])
    await expect(page.locator('#'+id)).not.toBeChecked();
  expect(app.writes.map(w=>w.path)).toEqual(['/api/mode','/api/connect','/api/mode']);
  await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await expect(page.locator('#control-mode')).toBeVisible();
  await page.screenshot({path:testInfo.outputPath('mode-selector-mobile.png'),fullPage:true});
  expect(app.errors).toEqual([]);
});

test('failed mode change keeps the current mode and displays the failure', async ({page}) => {
  const app=await setup(page,{fail:true});
  await page.locator('#control-mode').selectOption('hardware');
  await expect(page.locator('#control-mode-note')).toContainText('Mode change failed');
  await expect(page.locator('#control-mode')).toHaveValue('simulation');
  await expect(page.locator('#control-mode')).toBeEnabled();
  await expect(page.locator('.robot-toolbar')).toBeHidden();
  // A later periodic state refresh must not erase the failed mode change.
  await page.waitForTimeout(1100);
  await expect(page.locator('#control-mode-note')).toContainText('Mode change failed');
  expect(app.writes).toHaveLength(1); expect(app.errors).toEqual([]);
});

test('locked simulation launch disables the hardware option', async ({page}) => {
  const app=await setup(page,{locked:true});
  await expect(page.locator('#control-mode option[value="hardware"]')).toHaveJSProperty('disabled',true);
  await page.locator('#control-mode').focus();
  await page.keyboard.press('ArrowDown');
  await expect(page.locator('#control-mode')).toHaveValue('simulation');
  await expect(page.locator('#control-mode-note')).toContainText('Locked to simulation');
  expect(app.writes).toEqual([]); expect(app.errors).toEqual([]);
});

test('a mode change in another tab reloads and cancels the old tracking session', async ({page}) => {
  const app=await setup(page);
  await page.locator('#toggle-mouse-aim').click();
  await expect(page.locator('#toggle-mouse-aim')).toHaveAttribute('aria-pressed','true');
  app.state.hardware_enabled=true; app.state.control_mode='hardware';
  await ready(page,'hardware');
  await expect(page.locator('.robot-toolbar')).toBeVisible();
  await expect(page.locator('#toggle-mouse-aim')).toHaveAttribute('aria-pressed','false');
  await expect(page.locator('#toggle-mouse-aim')).toBeDisabled();
  expect(app.writes).toEqual([]); expect(app.errors).toEqual([]);
});
