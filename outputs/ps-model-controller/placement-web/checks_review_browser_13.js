// Run with the Playwright page-function runner against the local server.
async page => {
  const errors=[];
  page.on('pageerror', error=>errors.push(error.message));
  await page.goto('http://127.0.0.1:8890/review/13');
  await page.waitForFunction(()=>window.reviewState?.which==='new',null,{timeout:30000});
  await page.waitForTimeout(300);
  const canvas=page.locator('#canvas');
  const capture=()=>page.evaluate(()=>document.getElementById('canvas').toDataURL());
  const initial=await capture();
  await page.getByRole('button',{name:'原始模型'}).click();
  await page.waitForFunction(()=>window.reviewState?.which==='old');
  await page.waitForTimeout(300);
  const old=await capture();
  await page.getByRole('button',{name:'分部件重建 v1'}).click();
  await page.waitForFunction(()=>window.reviewState?.which==='new');
  await page.waitForTimeout(300);
  await page.getByRole('button',{name:'灰模'}).click();
  await page.waitForFunction(()=>window.reviewState?.mode==='clay');
  await page.waitForTimeout(200);
  const clay=await capture();
  await page.getByRole('button',{name:'材质',exact:true}).click();
  await page.waitForFunction(()=>window.reviewState?.mode==='material');
  await page.waitForTimeout(200);
  const material=await capture();
  const sample=await page.evaluate(()=>{
    const canvas=document.getElementById('canvas'),copy=document.createElement('canvas');
    copy.width=canvas.width;copy.height=canvas.height;
    const ctx=copy.getContext('2d');ctx.drawImage(canvas,0,0);
    const pixels=ctx.getImageData(0,0,copy.width,copy.height).data, background=[pixels[0],pixels[1],pixels[2]];
    const luma=[];
    for(let i=0;i<pixels.length;i+=4){
      const difference=Math.abs(pixels[i]-background[0])+Math.abs(pixels[i+1]-background[1])+Math.abs(pixels[i+2]-background[2]);
      if(difference>20)luma.push(.2126*pixels[i]+.7152*pixels[i+1]+.0722*pixels[i+2]);
    }
    luma.sort((a,b)=>a-b);
    return {pixels:luma.length,median:luma.length?luma[Math.floor(luma.length/2)]:0,
            dark_fraction:luma.length?luma.filter(v=>v<18).length/luma.length:1};
  });
  const box=await canvas.boundingBox();
  await page.mouse.move(box.x+box.width*.45,box.y+box.height*.5);
  await page.mouse.down();await page.mouse.move(box.x+box.width*.62,box.y+box.height*.55,{steps:8});await page.mouse.up();
  await page.waitForTimeout(400);
  const rotated=await capture();
  const links=await page.evaluate(()=>[...document.querySelectorAll('aside a')].map(a=>a.getAttribute('href')));
  const result={old_new_differ:initial!==old,clay_material_differ:clay!==material,
          pointer_rotates:rotated!==material,preview_luma:sample,links,
          final_state:await page.evaluate(()=>window.reviewState),errors};
  if(!result.old_new_differ||!result.clay_material_differ||!result.pointer_rotates||
     sample.pixels<100||sample.median<18||sample.dark_fraction>.8||errors.length)throw Error(JSON.stringify(result));
  if(result.final_state.which!=='new'||result.final_state.mode!=='material'||
     !String(result.final_state.id||'').endsWith('-rebuild-v1'))throw Error(JSON.stringify(result));
  if(!links.includes('/api/review/13/board')||!links.includes('/api/review/13/blend'))
    throw Error(JSON.stringify(result));
  return result;
}
