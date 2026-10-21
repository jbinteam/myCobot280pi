import {test, expect} from '@playwright/test';
const bridge=process.env.SIM_BRIDGE_URL||'http://127.0.0.1:8000';
test('workspace cloud, transparent volume and settings invalidation', async ({page}, testInfo) => {
  const errors=[]; page.on('pageerror', e=>errors.push(e.message));
  await page.route('**/api/**', async route => {
    const response=await route.fetch({url:`${bridge}${new URL(route.request().url()).pathname}`});
    await route.fulfill({response});
  });
  await page.goto('/?view=lab');
  await expect(page.locator('#run-results tbody tr')).toHaveCount(4, {timeout:45000});
  await page.locator('#workspace-samples').selectOption('4000');
  await page.locator('#workspace-targets').selectOption('64');
  let workspaceRequest;
  page.on('request', r => { if (new URL(r.url()).pathname === '/api/sim/workspace') workspaceRequest=r.postDataJSON(); });
  await page.locator('#generate-workspace').click();
  await expect(page.locator('#workspace-status')).toContainText('targets solved', {timeout:45000});
  const inspect=()=>page.evaluate(async()=>{
    const {scene}=await import('/src/main.js');
    const group=scene.getObjectByName('reachable-workspace');
    const cloud=scene.getObjectByName('reachable-workspace-cloud'), shell=scene.getObjectByName('reachable-workspace-volume');
    const fk=scene.getObjectByName('fk-workspace-cloud'), fkShell=scene.getObjectByName('fk-workspace-volume');
    return {size:cloud?.material.size,sizeAttenuation:cloud?.material.sizeAttenuation,order:cloud?.renderOrder,fkOrder:fk?.renderOrder,fkCount:fk?.geometry.attributes.position.count,fkVisible:fk?.visible,fkShellVisible:fkShell?.visible,shellVisible:shell?.visible,count:cloud?.geometry.attributes.position.count,faces:shell?.geometry.attributes.position.count,opacity:shell?.material.opacity,transparent:shell?.material.transparent,cloudVisible:cloud?.visible,children:group.children.length};
  });
  const result=await inspect(); expect(result.count).toBeGreaterThan(0); expect(workspaceRequest.start_deg).toEqual([0,-40,-130,80,0,50]); expect(workspaceRequest.method).toBe('bounded_dls'); expect(workspaceRequest.targets).toBe(64); expect(result.faces).toBeGreaterThan(0);
  expect(result.size).toBe(8); expect(result.sizeAttenuation).toBe(false); expect(result.order).toBeGreaterThan(result.fkOrder);
  await expect(page.locator('#solvable-cloud-count')).toHaveText(`(${result.count} points)`);
  expect(result.transparent).toBe(true); expect(result.opacity).toBeCloseTo(.12);
  expect(result.fkCount).toBeGreaterThan(result.count);
  await page.locator('#show-fk-workspace-cloud').uncheck();
  expect((await inspect()).fkVisible).toBe(false); expect((await inspect()).fkShellVisible).toBe(false); expect((await inspect()).cloudVisible).toBe(true);
  await page.locator('#show-fk-workspace-cloud').check();
  await page.locator('#workspace-opacity').fill('0.3'); expect((await inspect()).opacity).toBeCloseTo(.3);
  await page.locator('#show-workspace-cloud').uncheck(); expect((await inspect()).cloudVisible).toBe(false); expect((await inspect()).shellVisible).toBe(false); expect((await inspect()).fkVisible).toBe(true);
  await page.locator('#show-workspace-cloud').check();
  await page.screenshot({path:testInfo.outputPath('workspace.png'),fullPage:true});
  await page.locator('#ik-case').selectOption('1');
  await expect(page.locator('#workspace-status')).toContainText('showing the previous workspace');
  expect((await inspect()).count).toBe(result.count);
  expect((await inspect()).children).toBe(4);
  expect((await inspect()).cloudVisible).toBe(true);
  await page.locator('#target-rx').fill('30');
  expect((await inspect()).children).toBe(4);
  await expect(page.locator('#workspace-status')).toContainText('Generate again'); expect(errors).toEqual([]);
});

test('zero solver results explain why only FK is visible', async ({page}) => {
  await page.route('**/api/**', async route => {
    const response=await route.fetch({url:`${bridge}${new URL(route.request().url()).pathname}`});
    if (new URL(route.request().url()).pathname === '/api/sim/workspace') {
      const data=await response.json();
      await route.fulfill({json:{...data,accepted:0,points_mm:[],voxels:[],bounds_mm:null}});
    } else await route.fulfill({response});
  });
  await page.goto('/?view=lab');
  await expect(page.locator('#run-results tbody tr')).toHaveCount(4,{timeout:45000});
  await page.locator('#workspace-targets').selectOption('64');
  await page.locator('#workspace-samples').selectOption('4000');
  await page.locator('#generate-workspace').click();
  await expect(page.locator('#workspace-status')).toContainText('Solvable cloud is empty: 0 / 64',{timeout:45000});
  await expect(page.locator('#solvable-cloud-count')).toHaveText('(0 points)');
  await expect(page.locator('#fk-cloud-count')).not.toHaveText('(0 points)');
});
