// Keep position (mm) and direction (degrees) on separate, explicitly labeled plots.
export function renderErrorPlot(element, curves, {field, label, timeLabel='Simulation time (s)', minimum=1, height=235, signed=false}) {
  const points=curves.flatMap(c=>c.frames).filter(f=>Number.isFinite(f[field]));
  if(!points.length){element.replaceChildren();return;}
  const legend=curves.some(c=>c.label);
  const w=720,h=height,l=57,r=14,t=legend?35:18,b=40;
  let xmax=.05,peak=minimum;
  for(const point of points){xmax=Math.max(xmax,point.time);peak=Math.max(peak,signed?Math.abs(point[field]):point[field]);}
  const ymax=peak*1.05;
  const ymin=signed?-ymax:0;
  const x=v=>l+v/xmax*(w-l-r),y=v=>h-b-(v-ymin)/(ymax-ymin)*(h-t-b);
  let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${label} over ${timeLabel}">`;
  for(let i=0;i<=4;i++) {
    const xx=x(xmax*i/4),value=ymin+(ymax-ymin)*i/4,yy=y(value);
    svg+=`<path d="M${l},${yy}H${w-r} M${xx},${t}V${h-b}" stroke="#b4beb2" opacity=".5"/><text x="${l-7}" y="${yy+4}" text-anchor="end">${value.toFixed(ymax<.3?2:ymax<3?1:0)}</text><text x="${xx}" y="${h-b+17}" text-anchor="middle">${(xmax*i/4).toFixed(xmax<.1?3:xmax<1?2:1)}</text>`;
  }
  if(signed)svg+=`<path d="M${l},${y(0)}H${w-r}" stroke="#647267"/>`;
  if(legend)curves.forEach((c,i)=>{
    const left=l+i*(w-l-r)/curves.length;
    svg+=`<path d="M${left},14h18" stroke="${c.color}" stroke-width="2"/><text x="${left+24}" y="18">${c.label||''}</text>`;
  });
  for(const c of curves)svg+=`<polyline fill="none" stroke="${c.color}" stroke-width="2" points="${c.frames.filter(f=>Number.isFinite(f[field])).map(f=>`${x(f.time)},${y(f[field])}`).join(' ')}"/>`;
  svg+=`<text x="${w/2}" y="${h-3}" text-anchor="middle">${timeLabel}</text><text transform="translate(13 ${h/2}) rotate(-90)" text-anchor="middle">${label}</text></svg>`;
  element.innerHTML=svg;
}

export function renderJointVelocityPlot(element,points,options={}) {
  const colors=['#386baf','#30795c','#b75283','#8a652c','#795ac0','#478385'];
  renderErrorPlot(element,colors.map((color,j)=>({color,label:`J${j+1}`,
    frames:points.map(p=>({time:p.time,velocity:p.velocity_rad_s[j]}))})),
    {field:'velocity',label:'Joint velocities (rad/s)',minimum:.1,height:210,signed:true,...options});
}
