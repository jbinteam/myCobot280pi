const axes = ['x', 'y', 'z'];
export const offsetInputIds = axes.map(axis => `eef-offset-${axis}`);

// Full base-from-EEF rotation (row-major), including wrist spin.
export function addLocalOffset(base, offset, rotation) {
  return base.map((value, i) => value + rotation[i].reduce((sum, r, j) => sum + r * offset[j], 0));
}

export function readTargetFrame(rotation) {
  const read = id => {
    const value = document.getElementById(id).value.trim();
    return value === '' ? NaN : Number(value);
  };
  const base = axes.map(axis => read(`target-${axis}`));
  const offset = offsetInputIds.map(read);
  return { base, offset, rotation, target: addLocalOffset(base, offset, rotation) };
}

export function resetLocalOffset() {
  offsetInputIds.forEach(id => { document.getElementById(id).value = '0'; });
}
