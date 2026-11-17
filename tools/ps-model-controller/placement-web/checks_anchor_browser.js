// Run with playwright-cli run-code --filename after generating a preview.
// Uses the current pose, restores it afterwards, and never picks scene sites.
async page => {
  const read = () => page.evaluate(async () => {
    const path = document.getElementById('background').getAttribute('src');
    if (!path) throw Error('Choose a background and model first');
    return (await fetch(path.replace('/background',''))).json();
  });
  const input = async (id,value) => {
    const response = page.waitForResponse(r=>r.url().endsWith('/state')&&r.request().method()==='POST');
    await page.locator('#'+id).fill(String(value));
    await page.locator('#'+id).dispatchEvent('change');
    if (!(await response).ok()) throw Error('State update rejected');
  };
  const initial = await read(), anchor = initial.state.contact_anchor;
  if (!anchor) throw Error('Enable fixed contact reference first');
  const states = [initial.state];
  try {
    for (const [id,value] of [['azimuth',110],['elevation',35],['elevation',55],['size',900],['rotZ',35]]) {
      await input(id,value);
      const current = await read();
      if (Math.hypot(...current.state.contact_anchor.map((v,i)=>v-anchor[i]))>1e-6) {
        throw Error('Reference moved after changing '+id);
      }
      states.push(current.state);
    }
  } finally {
    for (const [id,key] of [['azimuth','azimuth'],['elevation','elevation'],['size','size'],['rotZ','rot_z']]) {
      await input(id,initial.state[key]);
    }
  }
  const restored = await read();
  if (Math.hypot(restored.state.x-initial.state.x, restored.state.y-initial.state.y)>1e-6) {
    throw Error('Restoring the pose did not restore the image position');
  }
  // Disabling the new mode retains the legacy centre-based interaction.
  await page.locator('#lockContact').uncheck();
  await input('azimuth', initial.state.azimuth+20);
  const unlocked = await read();
  if (unlocked.state.contact_anchor || unlocked.state.x!==restored.state.x || unlocked.state.y!==restored.state.y) {
    throw Error('Legacy centre placement changed');
  }
  await input('azimuth', initial.state.azimuth);
  const checked=page.waitForResponse(r=>r.url().endsWith('/state')&&r.request().method()==='POST');
  await page.locator('#lockContact').check(); await checked;
  await page.getByRole('button',{name:'拖动移动',exact:true}).click();
  const drag = async (selector,dx,dy,alt=false) => {
    const box = await page.locator(selector).boundingBox();
    const response=page.waitForResponse(r=>r.url().endsWith('/state')&&r.request().method()==='POST');
    if(alt) await page.keyboard.down('Alt');
    await page.mouse.move(box.x+box.width/2,box.y+box.height/2);
    await page.mouse.down();
    await page.mouse.move(box.x+box.width/2+dx,box.y+box.height/2+dy,{steps:3});
    await page.mouse.up();
    if(alt) await page.keyboard.up('Alt');
    if(!(await response).ok()) throw Error('Drag state rejected');
    return (await read()).state;
  };
  const moved=await drag('#object',18,-12);
  const scale=await page.evaluate(()=>document.getElementById('background').naturalWidth/document.getElementById('stage').clientWidth);
  const delta=moved.contact_anchor.map((v,i)=>v-anchor[i]);
  if(Math.hypot(delta[0]-18*scale,delta[1]+12*scale)>.1) throw Error('Dragging did not move the reference');
  await drag('#object',-18,12);
  for(const state of [await drag('#rotate',20,-10,true), await drag('#resize',15,10)]) {
    if(Math.hypot(...state.contact_anchor.map((v,i)=>v-anchor[i]))>.1) throw Error('Handle moved fixed reference');
    states.push(state);
  }
  for(const [id,key] of [['azimuth','azimuth'],['elevation','elevation'],['size','size']]) await input(id,initial.state[key]);
  const final=(await read()).state;
  if(Math.hypot(final.x-initial.state.x, final.y-initial.state.y)>.1) throw Error('Drag restoration failed');
  return {status:'passed', id:initial.id, original_pose_restored:true, drag_move_rotate_resize:true, states};
}
