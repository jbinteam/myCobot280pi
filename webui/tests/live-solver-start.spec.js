import {test,expect} from '@playwright/test';

test('each connected solve captures current robot joints despite case starts, preview edits, and playback',async({page,request})=>{
  const backend=await(await request.get('http://127.0.0.1:8000/api/state')).json();
  const info=await(await request.get('http://127.0.0.1:8000/api/sim/info')).json();
  let angles=[1.234,-40,-130,80,0,50],stale=false;
  const calls=[];
  await page.route('**/api/**',async route=>{
    const r=route.request(),path=new URL(r.url()).pathname;
    if(path==='/api/state'){
      await route.fulfill({json:{hardware_enabled:true,connected:true,token:backend.token,angles,updated:Date.now()/1000-(stale?30:0),motion:null,moving:false,limits:info.limits_deg,ik:{state:'idle',samples:[]}}});return;
    }
    if(path==='/api/sim/info'){await route.fulfill({json:info});return;}
    if(path==='/api/sim/solve'){
      const body=r.postDataJSON();calls.push(body);
      const response=await request.post('http://127.0.0.1:8000/api/sim/solve',{headers:{'X-Robot-Token':backend.token},data:body});
      await route.fulfill({status:response.status(),json:await response.json()});return;
    }
    throw new Error(`Unexpected robot request ${path}`);
  });
  await page.goto('/?view=lab');await expect(page.locator('#solve-ik')).toBeEnabled({timeout:45000});
  await page.locator('#follow-live').uncheck();
  await page.locator('#ik-case').selectOption('0');
  await page.locator('#compare-ik').click();
  await expect(page.locator('#ik-feedback')).toContainText('Start: live robot joints');
  expect(calls).toHaveLength(1);expect(calls[0].start_deg).toEqual(angles);expect(calls[0].methods).toHaveLength(info.pose5_methods.length);
  await page.locator('#ik-scrub').fill(await page.locator('#ik-scrub').getAttribute('max'));
  await page.locator('#rest-pose').click();
  angles=[4.567,-38,-128,78,2,53];
  await page.locator('#solve-ik').click();
  await expect(page.locator('#ik-feedback')).toContainText('4.567');
  expect(calls).toHaveLength(2);expect(calls[1].start_deg).toEqual(angles);
  stale=true;
  await page.locator('#solve-ik').click();
  await expect(page.locator('#ik-feedback')).toContainText('telemetry is unavailable or stale');
  expect(calls).toHaveLength(2);
});
