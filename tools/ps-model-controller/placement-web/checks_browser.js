// Run using playwright-cli run-code --filename <this file> after opening the page.
// This samples one neutral browser orientation, not full geometric fidelity.
async page => {
  const rows = await page.evaluate(async () => (await (await fetch('/api/assets')).json())
    .filter(item => !item.manual_only && item.available));
  await page.getByRole('button', {name: '使用现有监控背景'}).click();
  await page.locator('#object').waitFor({state: 'visible', timeout: 30000});
  const result = [];
  for (const row of rows) {
    if (await page.locator('#asset').inputValue() !== row.id) {
      const pending = page.waitForResponse(r => r.url().endsWith('/state') && r.request().method() === 'POST', {timeout: 30000});
      await page.locator('#asset').selectOption(row.id);
      await pending;
    }
    const metrics = await page.evaluate(() => {
      const canvas = document.getElementById('modelCanvas'), scratch = document.createElement('canvas');
      scratch.width = canvas.width; scratch.height = canvas.height;
      const ctx = scratch.getContext('2d'); ctx.drawImage(canvas, 0, 0);
      const pixels = ctx.getImageData(0, 0, scratch.width, scratch.height).data;
      let opaque = 0, dark = 0, total = 0;
      for (let i = 0; i < pixels.length; i += 4) if (pixels[i + 3] > 128) {
        opaque++; const light = (pixels[i] + pixels[i+1] + pixels[i+2]) / 3;
        total += light; if (light < 18) dark++;
      }
      return {opaque, mean_rgb: opaque ? Math.round(total/opaque) : 0,
              dark_fraction: opaque ? dark/opaque : 1};
    });
    result.push({number: row.number, ...metrics});
  }
  return result;
}
