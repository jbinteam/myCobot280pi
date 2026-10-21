const $ = id => document.getElementById(id);
export const directionInputIds = ['direction-entry', 'target-ux', 'target-uy', 'target-uz', 'direction-preset'];

export function normalizeToolAxis(values) {
  if (values.length !== 3 || !values.every(v => typeof v === 'number' && Number.isFinite(v))) throw new Error('Enter three finite tool-Z direction components in the base frame');
  const scale = Math.max(...values.map(Math.abs));
  if (!scale) throw new Error('Tool-Z direction must be a nonzero vector');
  const scaled = values.map(v => v / scale), length = Math.hypot(...scaled);
  return scaled.map(v => v / length);
}
export function axisFromTilts([rx, ry]) {
  if (![rx, ry].every(Number.isFinite)) throw new Error('Enter two finite tilt angles in degrees');
  const roll = rx * Math.PI / 180, pitch = ry * Math.PI / 180;
  return normalizeToolAxis([Math.sin(pitch) * Math.cos(roll), -Math.sin(roll), Math.cos(pitch) * Math.cos(roll)]);
}
export function tiltsFromAxis(axis) {
  const [x, y, z] = normalizeToolAxis(axis);
  return [Math.asin(Math.max(-1, Math.min(1, -y))) * 180 / Math.PI, Math.hypot(x, z) > 1e-12 ? Math.atan2(x, z) * 180 / Math.PI : 0];
}
const read = id => $(id).value.trim() === '' ? NaN : Number($(id).value);
export function readToolAxis() {
  return $('direction-entry').value === 'axis'
    ? normalizeToolAxis(['x', 'y', 'z'].map(a => read(`target-u${a}`)))
    : axisFromTilts(['target-rx', 'target-ry'].map(read));
}
export function readReferenceTilts() {
  return $('direction-entry').value === 'axis' ? tiltsFromAxis(readToolAxis()) : ['target-rx', 'target-ry'].map(read);
}
function writeAxis(axis) {
  normalizeToolAxis(axis).forEach((v, i) => { $(`target-u${'xyz'[i]}`).value = v.toFixed(8); });
}
export function syncAxisFromTilts() { writeAxis(axisFromTilts(['target-rx', 'target-ry'].map(read))); }

export function initToolDirection(getCurrentAxis) {
  function refresh() {
    const pose5 = $('ik-task').value === 'pose5', vector = $('direction-entry').value === 'axis';
    $('direction-entry-field').hidden = $('pose5-proposal-field').hidden = !pose5;
    $('orientation-fields').hidden = !pose5 || vector;
    $('direction-vector-fields').hidden = !pose5 || !vector;
    $('tool-direction-readout').hidden = !pose5;
    try {
      const axis = readToolAxis();
      $('target-tool-direction').textContent = `[${axis.map(v => v.toFixed(4)).join(', ')}]`;
      $('target-tool-direction').classList.remove('error');
    } catch (error) {
      $('target-tool-direction').textContent = error.message;
      $('target-tool-direction').classList.add('error');
    }
  }
  $('direction-entry').addEventListener('change', refresh);
  $('ik-task').addEventListener('change', refresh);
  for (const id of ['target-rx', 'target-ry']) $(id).addEventListener('input', () => {
    if ($('direction-entry').value === 'tilts') {
      try { syncAxisFromTilts(); } catch { /* Keep invalid inputs visible. */ }
    }
    refresh();
  });
  for (const id of ['target-ux', 'target-uy', 'target-uz']) $(id).addEventListener('input', () => {
    try {
      tiltsFromAxis(readToolAxis()).forEach((v, i) => {
        const input = $(`target-r${'xy'[i]}`); input.value = v.toFixed(8); input.dispatchEvent(new Event('input'));
      });
    } catch { /* Rejected before a solver/planner request. */ }
    refresh();
  });
  $('direction-preset').addEventListener('change', () => {
    const preset = $('direction-preset').value;
    if (!preset) return;
    const axis = preset === 'current' ? getCurrentAxis() : preset.split(',').map(Number);
    $('direction-entry').value = 'axis'; writeAxis(axis);
    $('direction-entry').dispatchEvent(new Event('change'));
    $('target-ux').dispatchEvent(new Event('input'));
    $('direction-preset').value = '';
  });
  refresh();
  return { refresh };
}
