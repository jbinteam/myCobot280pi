export function shapeSettings(value={kind:'quintic'}) {
  if(!value||typeof value!=='object'||Array.isArray(value))throw new Error('Invalid reference velocity shape');
  if(value.kind==='quintic'&&Object.keys(value).length===1)return {kind:'quintic'};
  const {kind,rise_percent:rise,fall_percent:fall}=value;
  if(kind!=='smooth_trapezoid'||Object.keys(value).sort().join(',')!=='fall_percent,kind,rise_percent'
    ||!Number.isFinite(rise)||!Number.isFinite(fall)||rise<5||rise>80||fall<5||fall>80||rise+fall>100)
    throw new Error('Acceleration and deceleration time must each be 5–80%, with a total at most 100%');
  return {kind,rise_percent:rise,fall_percent:fall};
}
export function shapePeaks(shape) {
  if(!shape||shape.kind==='quintic')return {rate:1.875,acceleration:10*Math.sqrt(3)/3};
  const rise=shape.rise_percent/100,fall=shape.fall_percent/100,rate=1/(1-(rise+fall)/2);
  return {rate,acceleration:1.5*rate/Math.min(rise,fall)};
}
export function shapeValue(u,shape) {
  u=Math.max(0,Math.min(1,u));
  if(!shape||shape.kind==='quintic')return {progress:10*u**3-15*u**4+6*u**5,rate:30*u**2*(1-u)**2,acceleration:60*u*(1-u)*(1-2*u)};
  const rise=shape.rise_percent/100,fall=shape.fall_percent/100,peak=1/(1-(rise+fall)/2);
  let progress,rate,acceleration;
  if(u<rise) {
    const x=u/rise;progress=peak*rise*(x**3-.5*x**4);rate=peak*(3*x**2-2*x**3);acceleration=peak*(6*x-6*x**2)/rise;
  }else if(u<=1-fall) {
    progress=peak*(u-rise/2);rate=peak;acceleration=0;
  }else {
    const w=(u-(1-fall))/fall;progress=peak*(1-fall-rise/2+fall*(w-w**3+.5*w**4));
    rate=peak*(1-3*w**2+2*w**3);acceleration=-peak*(6*w-6*w**2)/fall;
  }
  return {progress:u===1?1:progress,rate,acceleration};
}
const $=id=>document.getElementById(id);
export function readShapeEditor() {
  return shapeSettings($('velocity-shape').value==='quintic'?{kind:'quintic'}:
    {kind:'smooth_trapezoid',rise_percent:Number($('velocity-rise').value),fall_percent:Number($('velocity-fall').value)});
}
export function renderShapeEditor() {
  const adjustable=$('velocity-shape').value==='smooth_trapezoid';
  $('velocity-shape-ramps').hidden=!adjustable;
  $('velocity-rise-value').textContent=`${$('velocity-rise').value}%`;
  $('velocity-fall-value').textContent=`${$('velocity-fall').value}%`;
  try {
    const shape=readShapeEditor(),peak=shapePeaks(shape).rate;
    const points=Array.from({length:201},(_,i)=>`${30+300*i/200},${100-80*shapeValue(i/200,shape).rate/peak}`).join(' ');
    $('velocity-shape-preview').innerHTML=`<svg viewBox="0 0 360 130" role="img" aria-label="Reference velocity shape preview"><path d="M30 15V100H330" fill="none" stroke="#cbd5e1"/><polyline points="${points}" fill="none" stroke="#2465d6" stroke-width="2.5"/><g fill="#64748b" font-size="10"><text x="24" y="119">0%</text><text x="168" y="119">50%</text><text x="311" y="119">100%</text><text x="3" y="23">1</text><text x="3" y="103">0</text></g></svg>`;
    $('velocity-shape-feedback').textContent=adjustable?`${100-shape.rise_percent-shape.fall_percent}% of each leg at peak speed. Plan again to apply this shape to J1–J6.`:'Original bell-shaped profile. Plan again after changing the shape.';
    $('velocity-shape-feedback').classList.remove('error');
  }catch(error){
    $('velocity-shape-preview').replaceChildren();$('velocity-shape-feedback').textContent=error.message;$('velocity-shape-feedback').classList.add('error');
  }
}
