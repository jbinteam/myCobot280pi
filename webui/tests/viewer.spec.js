import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await page.route('**/api/sim/info', route => route.fulfill({json:{error:'Simulation disabled in viewer unit test'}}));
  await page.route('**/api/state', route => route.fulfill({json:{connected:false,token:'viewer-test'}}));
});

test('loads every model asset and updates the six-joint preview', async ({ page }, testInfo) => {
  const errors = [], failures = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('response', response => { if (response.status() >= 400) failures.push(response.url()); });
  await page.goto('/?view=lab');
  await expect(page.locator('#model-status')).toHaveText('Model ready', { timeout: 45000 });
  await expect(page.locator('#joint-controls input[type=range]')).toHaveCount(6);
  await expect(page.locator('.link-item')).toHaveCount(8);
  // Asset fetch success alone does not prove that its geometry was rendered.
  // In particular, joint1 and joint5 contain COLLADA <polygons> faces.
  const meshesPerLink = await page.evaluate(async () => {
    const { robot } = await import(document.querySelector('script[src*="/src/main.js"]').src);
    const counts = Object.fromEntries(Object.keys(robot.links).map(name => [name, 0]));
    robot.traverse(object => {
      if (!object.isMesh || !object.geometry.attributes.position?.count) return;
      let link = object.parent;
      while (link && !link.isURDFLink) link = link.parent;
      if (link) counts[link.name]++;
    });
    return counts;
  });
  expect(Object.keys(meshesPerLink)).toHaveLength(8);
  for (const [link, count] of Object.entries(meshesPerLink)) expect(count, `${link} has rendered geometry`).toBeGreaterThan(0);
  const rendered = await page.locator('canvas').evaluate(canvas => {
    const gl = canvas.getContext('webgl2');
    return { width: canvas.width, height: canvas.height, error: gl.getError() };
  });
  expect(rendered.width).toBeGreaterThan(300);
  expect(rendered.height).toBeGreaterThan(300);
  expect(rendered.error).toBe(0);
  await page.screenshot({ path: testInfo.outputPath('desktop.png'), fullPage: true });
  await page.locator('#rest-pose').click();
  await expect(page.locator('#pose-value')).toHaveText('[0,-40,-130,80,0,50]');
  const transformsAreFinite = await page.evaluate(async () => {
    const { robot, camera } = await import(document.querySelector('script[src*="/src/main.js"]').src);
    return camera.position.toArray().every(Number.isFinite) && Object.values(robot.joints).every(joint => [...joint.quaternion.toArray(), ...joint.axis.toArray()].every(Number.isFinite));
  });
  expect(transformsAreFinite).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('rest.png'), fullPage: true });
  await page.locator('#zero-pose').click();
  await expect(page.locator('#pose-value')).toHaveText('[0,0,0,0,0,0]');
  await page.locator('#joint-0-number').fill('45');
  await page.locator('#joint-0-number').press('Tab');
  await expect(page.locator('#pose-value')).toHaveText('[45,0,0,0,0,0]');
  await page.locator('#joint-1-number').fill('999');
  await page.locator('#joint-1-number').press('Tab');
  expect(Number(await page.locator('#joint-1-number').inputValue())).toBeLessThan(141);
  await page.locator('#joint-2-slider').focus();
  await page.locator('#joint-2-slider').press('ArrowRight');
  await expect(page.locator('#joint-2-number')).toHaveValue('0.1');
  for (const view of ['front', 'side', 'top', 'reset']) await page.locator(`#${view}-view`).click();
  expect(await page.evaluate(async () => {
    const { camera } = await import(document.querySelector('script[src*="/src/main.js"]').src);
    return camera.position.toArray().every(Number.isFinite);
  })).toBe(true);
  for (const id of ['show-grid', 'show-axes', 'wireframe']) {
    await page.locator(`#${id}`).click();
    await page.locator(`#${id}`).click();
  }
  expect(errors).toEqual([]);
  expect(failures).toEqual([]);
});

test('mobile layout stays within screen width', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/?view=lab');
  await expect(page.locator('#model-status')).toHaveText('Model ready', { timeout: 45000 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({ path: testInfo.outputPath('mobile.png'), fullPage: true });
});

test('missing model asset displays an error', async ({ page }, testInfo) => {
  await page.route('**/joint2.dae', route => route.fulfill({ status: 404, body: 'Not found' }));
  await page.goto('/?view=lab');
  await expect(page.locator('#model-status')).toHaveText('Model loading failed', { timeout: 45000 });
  await expect(page.locator('#loading-message')).toContainText('joint2.dae');
  await expect(page.locator('#loading-overlay')).not.toHaveClass('hidden');
});
