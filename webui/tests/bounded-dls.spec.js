import { test, expect } from '@playwright/test';
import { execFileSync } from 'node:child_process';

// Use fresh local code without changing the user's running hardware bridge.
function python(code, input='') {
  return JSON.parse(execFileSync('.venv/bin/python',['-c',code],{
    cwd:'..',input,encoding:'utf8',env:{...process.env,OPENBLAS_NUM_THREADS:'1'},
  }));
}

test('bounded DLS appears in the comparison with a colored result and converges',async({page})=>{
  // cwd is project root: the interpreter path is rooted there as well.
  const info=python('import json; from simulation.service import info; print(json.dumps(info()))');
  const errors=[],calls=[];
  page.on('pageerror',error=>errors.push(error.message));
  await page.route('**/api/**',async route=>{
    const request=route.request(),path=new URL(request.url()).pathname;
    if(path==='/api/sim/info')return route.fulfill({json:info});
    if(path==='/api/state')return route.fulfill({json:{hardware_enabled:false,connected:false,token:'bounded-ui',motion:null,moving:false,ik:{state:'idle',samples:[]}}});
    if(path==='/api/sim/solve') {
      calls.push(request.postDataJSON());
      const data=python('import json,sys; from simulation.service import run; print(json.dumps(run(json.load(sys.stdin))))',request.postData());
      return route.fulfill({json:data});
    }
    throw new Error(`Unexpected endpoint ${path}`);
  });
  await page.goto('/?view=lab');
  await expect(page.locator('#run-results tbody tr')).toHaveCount(4,{timeout:45000});
  await expect(page.locator('#joint-3-number')).toHaveAttribute('min','-143');
  await expect(page.locator('#joint-3-number')).toHaveAttribute('max','143');
  await expect(page.locator('.joint-control').nth(3).locator('.joint-limits')).toContainText('-143°');
  await expect(page.locator('.joint-control').nth(3).locator('.joint-limits')).toContainText('143°');
  await expect(page.locator('#compare-ik')).toHaveText('Compare all 4');
  await page.locator('#ik-method').selectOption('bounded_dls');
  await expect(page.locator('#ik-description')).toContainText('Joint-limit-aware');
  await expect(page.locator('#ik-summary')).toContainText('Reached target');
  await page.locator('#ik-case').selectOption('7');
  await page.locator('#solve-ik').click();
  await expect(page.locator('#ik-summary')).toContainText('Reached target');
  expect(calls.at(-1).methods).toEqual(['bounded_dls']);
  expect(await page.locator('#chart-legend button').evaluate(el=>el.style.borderLeftColor)).toBeTruthy();
  expect(errors).toEqual([]);
});
