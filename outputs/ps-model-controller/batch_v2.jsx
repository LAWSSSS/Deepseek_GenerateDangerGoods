/* Commands in JOB are assembled by batch_v2.py. */
function importLayer(doc,path) {
    var source=app.open(new File(path));
    try { return source.activeLayer.duplicate(doc,ElementPlacement.PLACEATBEGINNING); }
    finally { source.close(SaveOptions.DONOTSAVECHANGES);app.activeDocument=doc; }
}
function bounds(layer) {
    var b=layer.bounds;
    return [b[0].as("px"),b[1].as("px"),b[2].as("px"),b[3].as("px")];
}
function moveTo(layer,x,y) {
    var b=bounds(layer);
    layer.translate(x-b[0],y-b[1]);
}
function selectAlpha() {
    var c=charIDToTypeID,d=new ActionDescriptor(),s=new ActionReference(),t=new ActionReference();
    s.putProperty(c("Chnl"),c("fsel"));d.putReference(c("null"),s);
    t.putEnumerated(c("Chnl"),c("Chnl"),c("Trsp"));d.putReference(c("T   "),t);
    executeAction(c("setd"),d,DialogModes.NO);
}
function importPreparedAsset(doc,job) {
    if(!job.asset_trim_mask)return importLayer(doc,job.asset);
    var source=app.open(new File(job.asset));
    try {
        var object=source.activeLayer;
        var trim=importLayer(source,job.asset_trim_mask);
        source.activeLayer=trim;
        selectAlpha();
        trim.remove();
        source.activeLayer=object;
        source.selection.invert();
        source.selection.clear();
        source.selection.deselect();
        return object.duplicate(doc,ElementPlacement.PLACEATBEGINNING);
    } finally { source.close(SaveOptions.DONOTSAVECHANGES);app.activeDocument=doc; }
}
function maskFromSelection() {
    var c=charIDToTypeID,d=new ActionDescriptor(),r=new ActionReference();
    d.putClass(c("Nw  "),c("Chnl"));
    r.putEnumerated(c("Chnl"),c("Chnl"),c("Msk "));d.putReference(c("At  "),r);
    d.putEnumerated(c("Usng"),c("UsrM"),c("RvlS"));
    executeAction(c("Mk  "),d,DialogModes.NO);
}
function adjustment(doc,type,params,name) {
    var c=charIDToTypeID, d=new ActionDescriptor(), r=new ActionReference(), layer=new ActionDescriptor();
    r.putClass(c("AdjL"));d.putReference(c("null"),r);
    layer.putString(c("Nm  "),name);layer.putObject(c("Type"),c(type),params);
    d.putObject(c("Usng"),c("AdjL"),layer);executeAction(c("Mk  "),d,DialogModes.NO);
    doc.activeLayer.grouped=true;
}
function saturation(doc,value,name) {
    var c=charIDToTypeID,d=new ActionDescriptor(),l=new ActionList(),m=new ActionDescriptor();
    m.putInteger(c("H   "),0);m.putInteger(c("Strt"),Math.round(value));m.putInteger(c("Lght"),0);
    l.putObject(c("Hst2"),m);d.putList(c("Adjs"),l);
    adjustment(doc,"HStr",d,name);
}
function curveChannel(list,channel,delta,points) {
    var c=charIDToTypeID,d=new ActionDescriptor(),r=new ActionReference(),p=new ActionList();
    r.putEnumerated(c("Chnl"),c("Chnl"),c(channel));d.putReference(c("Chnl"),r);
    var values=points || [[0,0],[48,48+delta*.48],[128,128+delta],[208,208+delta*.48],[255,255]];
    for(var i=0;i<values.length;i++) {
        var point=new ActionDescriptor();
        point.putDouble(c("Hrzn"),values[i][0]);point.putDouble(c("Vrtc"),values[i][1]);
        p.putObject(c("Pnt "),point);
    }
    d.putList(c("Crv "),p);list.putObject(c("CrvA"),d);
}
function localPhoto(doc,t) {
    if(!t.luma_curve && !t.rgb_cast)return;
    var c=charIDToTypeID,d=new ActionDescriptor(),channels=new ActionList();
    if(t.luma_curve)curveChannel(channels,"Cmps",0,t.luma_curve);
    if(t.rgb_cast) {
        curveChannel(channels,"Rd  ",t.rgb_cast[0]);
        curveChannel(channels,"Grn ",t.rgb_cast[1]);
        curveChannel(channels,"Bl  ",t.rgb_cast[2]);
    }
    d.putList(c("Adjs"),channels);adjustment(doc,"Crvs",d,"Local exposure contrast and color cast");
}
function tone(doc,brightness,warmth) {
    var c=charIDToTypeID,d=new ActionDescriptor(),channels=new ActionList();
    if(brightness)curveChannel(channels,"Cmps",brightness);
    if(warmth){curveChannel(channels,"Rd  ",warmth);curveChannel(channels,"Bl  ",-warmth);}
    d.putList(c("Adjs"),channels);adjustment(doc,"Crvs",d,"Ambient light and reflected color");
}
function savePsd(doc,path) {
    var o=new PhotoshopSaveOptions();o.layers=true;o.embedColorProfile=true;
    doc.saveAs(new File(path),o,true);
}
function png(doc,path) {doc.saveAs(new File(path),new PNGSaveOptions(),true);}
function exportObjectAlpha(doc,object,path,rect) {
    var b=bounds(object);
    if(b[0]<rect[0] || b[1]<rect[1] || b[2]>rect[2] || b[3]>rect[3])
        throw Error("Object alpha exceeds preview crop");
    var alpha=app.documents.add(UnitValue(rect[2]-rect[0],"px"),UnitValue(rect[3]-rect[1],"px"),
                                doc.resolution,"Object alpha",NewDocumentMode.RGB,DocumentFill.TRANSPARENT);
    try {
        app.activeDocument=doc;
        var layer=object.duplicate(alpha,ElementPlacement.PLACEATBEGINNING);
        app.activeDocument=alpha;
        moveTo(layer,b[0]-rect[0],b[1]-rect[1]);
        png(alpha,path);
    } finally {alpha.close(SaveOptions.DONOTSAVECHANGES);app.activeDocument=doc;}
}
function preview(doc,path,rect,edge) {
    var p=doc.duplicate("Batch quality preview",true);
    try {
        p.crop([UnitValue(rect[0],"px"),UnitValue(rect[1],"px"),UnitValue(rect[2],"px"),UnitValue(rect[3],"px")]);
        var w=p.width.as("px"),h=p.height.as("px"),scale=Math.min(1,edge/Math.max(w,h));
        if(scale<1)p.resizeImage(UnitValue(Math.round(w*scale),"px"),UnitValue(Math.round(h*scale),"px"),
                                 null,ResampleMethod.BICUBIC);
        var o=new JPEGSaveOptions();o.quality=7;o.embedColorProfile=true;
        p.saveAs(new File(path),o,true);
    } finally {p.close(SaveOptions.DONOTSAVECHANGES);app.activeDocument=doc;}
}
function makeAsset(job) {
    var doc=app.open(new File(job.matte));
    try {
        var asset=doc.activeLayer;
        if(asset.isBackgroundLayer)throw Error("Asset matte has no transparency");
        asset.name="Photographed object";
        saturation(doc,job.preworn ? -4 : -15,"Faded finish");
        if(!job.preworn)tone(doc,-16,1);
        doc.activeLayer=asset;
        var dust=asset.duplicate(asset,ElementPlacement.PLACEBEFORE);
        doc.activeLayer=dust;dust.name="Dust on the existing surface";dust.desaturate();
        dust.adjustLevels(0,255,1.06,27,218);dust.opacity=job.preworn ? 6 : 18;
        dust.grouped=true;
        var grain=importLayer(doc,job.wear);
        doc.activeLayer=grain;grain.name="Weathering from photographed surface detail";
        grain.blendMode=BlendMode.NORMAL;grain.opacity=job.preworn ? 11 : 78;
        savePsd(doc,job.psd);png(doc,job.png);
        return {asset:true};
    } finally {doc.close(SaveOptions.DONOTSAVECHANGES);}
}
function addForeground(doc,background,entry,x0,y0,area) {
    if(!entry)return null;
    var marker=importLayer(doc,entry.path);
    doc.activeLayer=marker;moveTo(marker,entry.x-x0,entry.y-y0);selectAlpha();
    marker.remove();
    if(!area) {
        var legacy=background.duplicate(doc.layers[0],ElementPlacement.PLACEBEFORE);
        doc.activeLayer=legacy;legacy.name="Foreground scrap - original pixels";
        maskFromSelection();doc.selection.deselect();
        return legacy;
    }
    var l=Math.floor(area[0])-2,t=Math.floor(area[1])-2;
    var r=Math.ceil(area[2])+2,b=Math.ceil(area[3])+2;
    doc.selection.select([[l,t],[r,t],[r,b],[l,b]],SelectionType.INTERSECT,0,false);
    doc.activeLayer=background;
    executeAction(charIDToTypeID("CpTL"),undefined,DialogModes.NO);
    var overlay=doc.activeLayer;
    if(overlay.isBackgroundLayer)throw Error("Foreground copy did not create a layer");
    if(doc.layers.length>2)overlay.move(doc.layers[0],ElementPlacement.PLACEBEFORE);
    overlay.name="Foreground scrap - original pixels only";
    doc.selection.deselect();
    return overlay;
}
function groupObject(doc) {
    var content=[];
    for(var i=doc.layers.length-1;i>=0;i--) {
        var layer=doc.layers[i];
        if(!layer.isBackgroundLayer)content.push({layer:layer,clipped:layer.grouped});
    }
    for(var i=0;i<content.length;i++)if(content[i].clipped)content[i].layer.grouped=false;
    var group=doc.layerSets.add();group.name="Object, contact shadow and local match";
    var previous=null;
    for(var i=0;i<content.length;i++) {
        var layer=content[i].layer;
        if(previous)layer.move(previous,ElementPlacement.PLACEBEFORE);
        else layer.move(group,ElementPlacement.INSIDE);
        if(content[i].clipped)layer.grouped=true;
        previous=layer;
    }
}
function makeCandidate(source,job) {
    var doc=source.duplicate("Scrap composite candidate",false);
    try {
        if(doc.width.as("px")!==job.canvas[0] || doc.height.as("px")!==job.canvas[1])throw Error("Canvas size changed");
        var rect=job.crop,x0=0,y0=0;
        if(job.patch_png || (job.object_alpha && !job.psd && !job.png)) {
            // Preview at native scale with the same source pixels and layer offsets as the full render.
            doc.crop([UnitValue(rect[0],"px"),UnitValue(rect[1],"px"),UnitValue(rect[2],"px"),UnitValue(rect[3],"px")]);
            x0=rect[0];y0=rect[1];rect=[0,0,rect[2]-x0,rect[3]-y0];
        }
        var base=doc.backgroundLayer,shadeBounds=null;
        if(job.shade){
            var shadow=importLayer(doc,job.shade.path);
            doc.activeLayer=shadow;moveTo(shadow,job.shade.x-x0,job.shade.y-y0);
            shadow.name="Contact shadow on support";shadow.blendMode=BlendMode.MULTIPLY;shadow.opacity=55;
            shadeBounds=bounds(shadow);
        }
        if(job.contact_delta){
            var contact=importLayer(doc,job.contact_delta.path);
            doc.activeLayer=contact;
            moveTo(contact,job.contact_delta.x-x0,job.contact_delta.y-y0);
            contact.name="Local generated contact light and dust";
            var contactBounds=bounds(contact);
            if(!shadeBounds)shadeBounds=contactBounds;
            else for(var j=0;j<4;j++)
                shadeBounds[j]=(j<2?Math.min:Math.max)(shadeBounds[j],contactBounds[j]);
        }
        if(job.unmatched_asset){
            var unmatched=importLayer(doc,job.unmatched_asset);
            var originalBounds=bounds(unmatched);
            moveTo(unmatched,job.center[0]-x0-(originalBounds[2]-originalBounds[0])/2,
                             job.center[1]-y0-(originalBounds[3]-originalBounds[1])/2);
            unmatched.name="Independent object before color match";
            unmatched.visible=false;
        }
        var object=importPreparedAsset(doc,job);
        doc.activeLayer=object;object.name=job.unmatched_asset ? "Object matched to local steel" : "Weathered photographed object";
        if(job.angle)object.rotate(job.angle,AnchorPosition.MIDDLECENTER);
        for(var i=0;i<2;i++){
            var b=bounds(object),ratio=100*job.width/(b[2]-b[0]);
            if(Math.abs(ratio-100)>1)object.resize(ratio,ratio,AnchorPosition.MIDDLECENTER);
        }
        var b=bounds(object);
        moveTo(object,job.center[0]-x0-(b[2]-b[0])/2,job.center[1]-y0-(b[3]-b[1])/2);
        b=bounds(object);
        if(Math.abs(b[2]-b[0]-job.width)>Math.max(4,job.width*.04))throw Error("Unexpected Photoshop object width");
        var t=job.tone;
        var visible=object;
        if(t.blur_px){
            var softened=object.duplicate(object,ElementPlacement.PLACEBEFORE);
            doc.activeLayer=softened;softened.name="Match CCTV sharpness";
            softened.applyGaussianBlur(t.blur_px);object.visible=false;
            visible=softened;
        }
        if(job.object_alpha)exportObjectAlpha(doc,visible,job.object_alpha,rect);
        localPhoto(doc,t);
        if(t.brightness || t.warmth)tone(doc,t.brightness,t.warmth);
        if(t.saturation)saturation(doc,t.saturation,"Match reflected color saturation");
        if(job.grain) {
            var texture=importLayer(doc,job.grain);
            doc.activeLayer=texture;moveTo(texture,job.crop[0]-x0,job.crop[1]-y0);
            texture.name="CCTV texture from local background";
            texture.blendMode=BlendMode.SOFTLIGHT;texture.opacity=65;texture.grouped=true;
        }
        if(job.foreground_shadow) {
            var lip=importLayer(doc,job.foreground_shadow.path);
            doc.activeLayer=lip;
            moveTo(lip,job.foreground_shadow.x-x0,job.foreground_shadow.y-y0);
            lip.name="Shadow cast by foreground steel";lip.blendMode=BlendMode.MULTIPLY;
            lip.opacity=45;lip.grouped=true;
        }
        var area=null;
        if(job.foreground_alpha) {
            area=bounds(visible);
            if(shadeBounds)for(var i=0;i<4;i++)area[i]=(i<2?Math.min:Math.max)(area[i],shadeBounds[i]);
            if(job.psd)groupObject(doc);
        }
        var foreground=addForeground(doc,base,job.occlusion,x0,y0,area);
        if(job.foreground_alpha && foreground)exportObjectAlpha(doc,foreground,job.foreground_alpha,rect);
        if(job.psd)savePsd(doc,job.psd);
        if(job.png)png(doc,job.png);
        if(job.patch_png)png(doc,job.patch_png);
        if(job.preview)preview(doc,job.preview,rect,512);
        var result={bounds:[b[0]+x0,b[1]+y0,b[2]+x0,b[3]+y0]};
        if(area && foreground){
            var f=bounds(foreground);result.foreground_bounds=[f[0]+x0,f[1]+y0,f[2]+x0,f[3]+y0];
        }
        return result;
    } finally {doc.close(SaveOptions.DONOTSAVECHANGES);}
}
function runBatch(job) {
    var prior=app.documents.length ? app.activeDocument : null;
    var dialogs=app.displayDialogs,units=app.preferences.rulerUnits,source=null;
    try {
        app.displayDialogs=DialogModes.NO;app.preferences.rulerUnits=Units.PIXELS;
        if(job.mode==="asset")return makeAsset(job);
        source=app.open(new File(job.source));
        if(job.mode==="reference"){png(source,job.png);return {reference:true};}
        var results=[];
        for(var i=0;i<job.candidates.length;i++) {
            try {results.push(makeCandidate(source,job.candidates[i]));}
            catch(error){results.push({error:String(error.message || error)});}
        }
        return results;
    } finally {
        if(source)source.close(SaveOptions.DONOTSAVECHANGES);
        if(prior)app.activeDocument=prior;
        app.displayDialogs=dialogs;app.preferences.rulerUnits=units;
    }
}
JSON.stringify(runBatch(JOB));
