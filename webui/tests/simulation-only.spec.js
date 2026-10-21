import { test, expect } from '@playwright/test';

test('simulation-only default tracks a cursor through the real v2 API without hardware requests', async ({ page }, testInfo) => {
  const posts = [], errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/api/**', async route => {
    const request = route.request(), path = new URL(request.url()).pathname;
    if (request.method() === 'POST') {
      posts.push({ path, body: request.postDataJSON() });
      if (!path.startsWith('/api/sim/')) {
        await route.abort();
        throw new Error(`Simulation attempted hardware request: ${path}`);
      }
    }
    await route.continue();
  });
  await page.goto('/');
  await expect(page.locator('#mode-badge')).toContainText('Simulation only');
  await expect(page.locator('#recommended-model')).toHaveText('Local IK + Diffusion v2');
  await expect(page.locator('#toggle-mouse-aim')).toBeEnabled({ timeout: 45000 });
  await expect(page.locator('#simulation-setup')).toBeVisible();
  await expect(page.locator('.robot-toolbar')).toBeHidden();
  await expect(page.locator('#mouse-aim-planner')).toHaveValue('local_diffusion');
  await expect(page.locator('#mouse-aim-continuous')).toBeChecked();
  expect(posts).toEqual([]);
  await page.locator('#toggle-mouse-aim').click();
  await page.locator('#viewport canvas').scrollIntoViewIfNeeded();
  const point = await page.evaluate(async () => {
    const { scene, camera, renderer } = await import('/src/main.js');
    renderer.setAnimationLoop(null);
    const anchor = scene.getObjectByName('mouse-aim-cursor').position.clone();
    const offset = anchor.clone().set(.004, .002, .001);
    const normal = offset.clone().cross(anchor.clone().set(0, 0, 1)).normalize();
    camera.position.copy(anchor).addScaledVector(normal, .7);
    camera.lookAt(anchor); camera.updateMatrixWorld(true);
    const projected = anchor.add(offset).project(camera);
    const bounds = document.querySelector('#viewport canvas').getBoundingClientRect();
    window.simTestTime = performance.now();
    return { x: bounds.left+(projected.x+1)*bounds.width/2, y: bounds.top+(1-projected.y)*bounds.height/2 };
  });
  const responsePromise = page.waitForResponse(response => new URL(response.url()).pathname === '/api/sim/mouse-aim');
  await page.mouse.move(point.x, point.y);
  await page.evaluate(async () => {
    const { mouseAim } = await import('/src/main.js');
    mouseAim.update(window.simTestTime += 50);
  });
  const response = await responsePromise;
  expect(response.ok()).toBe(true);
  const plan = await response.json();
  expect(plan.converged).toBe(true);
  expect(plan.continuous_motion).toBe(true);
  expect(plan.smoothing.method).toBe('collision_checked_bezier_c2');
  await expect.poll(() => page.evaluate(async () => {
    const { mouseAim } = await import('/src/main.js');
    return Number.isFinite(mouseAim.getPreviewMotion().time);
  })).toBe(true);
  await page.evaluate(async () => {
    const { mouseAim, scene, camera, renderer } = await import('/src/main.js');
    window.simTestTime = Math.max(window.simTestTime, performance.now());
    for (let i = 0; i < 200; i++) mouseAim.update(window.simTestTime += 50);
    renderer.render(scene, camera);
  });
  await expect(page.locator('#mouse-aim-status')).toContainText('On target');
  expect(posts).toHaveLength(1);
  expect(posts[0]).toMatchObject({ path: '/api/sim/mouse-aim', body: {
    tracking_engine: 'local_diffusion', continuous_motion: true, obstacle_mm: null,
  } });
  await page.screenshot({ path: testInfo.outputPath('simulation-only-tracking.png'), fullPage: true });
  await page.keyboard.press('Escape');
  await expect(page.locator('#toggle-mouse-aim')).toHaveAttribute('aria-pressed', 'false');
  expect(errors).toEqual([]);
});
