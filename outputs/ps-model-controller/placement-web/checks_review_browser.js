// Run with playwright-cli run-code --filename <this file> against the local server.
async page => {
  const errors=[];
  page.on('pageerror', error=>errors.push(error.message));
  await page.goto('http://127.0.0.1:8765/review/01');
  await page.waitForFunction(()=>window.reviewState?.which==='new',null,{timeout:30000});
  await page.waitForTimeout(300);
  const canvas=page.locator('#canvas');
  const capture=()=>page.evaluate(()=>document.getElementById('canvas').toDataURL());
  const initial=await capture();
  await page.getByRole('button',{name:'旧版模型'}).click();
  await page.waitForFunction(()=>window.reviewState?.which==='old');
  await page.waitForTimeout(300);
  const old=await capture();
  await page.getByRole('button',{name:'新版重建'}).click();
  await page.waitForFunction(()=>window.reviewState?.which==='new');
  await page.getByRole('button',{name:'灰模'}).click();
  await page.waitForFunction(()=>window.reviewState?.mode==='clay');
  await page.waitForTimeout(200);
  const clay=await capture();
  await page.getByRole('button',{name:'材质',exact:true}).click();
  await page.waitForFunction(()=>window.reviewState?.mode==='material');
  await page.waitForTimeout(200);
  const material=await capture();
  const box=await canvas.boundingBox();
  await page.mouse.move(box.x+box.width*.45,box.y+box.height*.5);
  await page.mouse.down();await page.mouse.move(box.x+box.width*.62,box.y+box.height*.55,{steps:8});await page.mouse.up();
  await page.waitForTimeout(400);
  const rotated=await capture();
  return {old_new_differ:initial!==old,clay_material_differ:clay!==material,
          pointer_rotates:rotated!==material,final_state:await page.evaluate(()=>window.reviewState),errors};
}
