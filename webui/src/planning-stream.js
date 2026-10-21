export async function readPlanningStream(response,signal,onSearch) {
  if(!response.ok){const result=await response.json();throw new Error(result.error||`Planner HTTP ${response.status}`);}
  const reader=response.body.getReader(),decoder=new TextDecoder();
  let buffered='',result;
  function line(text) {
    if(!text.trim())return;
    if(signal.aborted)throw new DOMException('Approach interrupted','AbortError');
    const message=JSON.parse(text);
    if(message.type==='rrt')onSearch(message.events);
    else if(message.type==='result')result=message.result;
    else if(message.type==='error')throw new Error(message.error);
    else throw new Error('Unknown planning stream message');
  }
  try {
    while(true) {
      const {done,value}=await reader.read();
      buffered+=decoder.decode(value,{stream:!done});
      let end;
      while((end=buffered.indexOf('\n'))!==-1){line(buffered.slice(0,end));buffered=buffered.slice(end+1);}
      if(buffered.length>4000000)throw new Error('Planning stream line is too large');
      if(done){line(buffered);break;}
    }
    if(!result)throw new Error('Planning stream ended without a checked result');
    return result;
  } finally {await reader.cancel().catch(()=>{});reader.releaseLock();}
}
