import test from 'node:test';
import assert from 'node:assert/strict';
import {renderJointVelocityPlot} from '../src/ik-plot.js';
test('long route velocity chart renders beyond JavaScript argument-count limits',()=>{
 const element={innerHTML:'',replaceChildren(){this.innerHTML='';}};
 const points=Array.from({length:30000},(_,i)=>({time:i*.02,velocity_rad_s:[.1,-.1,.2,-.2,0,.05]}));
 renderJointVelocityPlot(element,points);
 assert.equal((element.innerHTML.match(/<polyline /g)||[]).length,6);
 assert.ok(element.innerHTML.includes('J6'));
 assert.ok(!element.innerHTML.includes('NaN'));
});
