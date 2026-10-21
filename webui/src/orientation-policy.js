// The effective solver is explicit; camera relaxation requires failed strict search.
export function effectiveOrientationMode(result,requested='point_at_object') {
  const mode=result.orientation_mode??'point_at_object';
  if(requested!=='strict_then_fov') {
    if(mode!==requested)throw new Error('Planner changed the requested orientation mode');
    return mode;
  }
  const selection=result.orientation_selection;
  if(result.requested_orientation_mode!==requested||selection?.policy!==requested||
      selection.selected_mode!==mode||selection.strict?.mode!=='point_at_object')
    throw new Error('Planner returned no checked strict-first orientation selection');
  if(mode==='point_at_object'&&selection.fallback_used===false&&selection.relaxed===null)
    return mode;
  if(mode==='camera_fov'&&selection.fallback_used===true&&
      ['ik_failed','no_path'].includes(selection.strict.status)&&selection.strict.playable===false&&
      selection.relaxed?.mode==='camera_fov')return mode;
  throw new Error('Camera-FOV fallback requires a failed strict search');
}
