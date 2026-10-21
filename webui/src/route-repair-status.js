export function routeRepairStatus(result) {
  const repair = result.guided_repair;
  if (!repair?.enabled) return 'Guided repair: off';
  if (!repair.triggered) return `Guided repair: ${result.converged ? 'not needed' : 'not attempted'}`;
  const outcome = repair.recovered
    ? (result.converged ? 'route recovered' : 'route found; preview validation failed')
    : 'no route found';
  return `Guided repair: ${outcome} · ${repair.new_nodes || 0} route samples${Number.isFinite(repair.repair_ms) ? ` · ${Math.round(repair.repair_ms)} ms` : ''}`;
}
