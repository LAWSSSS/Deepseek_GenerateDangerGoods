/* JOB supplies local file paths, a pixel crop and a validated placement plan. */
function placeAsset(job) {
    var oldUnits=app.preferences.rulerUnits;
    var dialogs=app.displayDialogs;
    var prior=app.documents.length ? app.activeDocument : null;
    var doc=null, cutout=null;
    try {
        app.preferences.rulerUnits=Units.PIXELS;
        app.displayDialogs=DialogModes.NO;
        doc=app.open(new File(job.background));
        var base=doc.backgroundLayer;
        var canvasWidth=doc.width.as("px"), canvasHeight=doc.height.as("px");
        var crop=job.crop, p=job.placement;
        if (crop[0]<0 || crop[1]<0 || crop[2]<=0 || crop[3]<=0 ||
            crop[0]+crop[2]>canvasWidth || crop[1]+crop[3]>canvasHeight) throw Error("Invalid crop");
        cutout=app.open(new File(job.asset));
        if (cutout.activeLayer.isBackgroundLayer) throw Error("Expected transparent cutout");
        var object=cutout.activeLayer.duplicate(doc,ElementPlacement.PLACEATBEGINNING);
        cutout.close(SaveOptions.DONOTSAVECHANGES);cutout=null;
        app.activeDocument=doc;
        doc.activeLayer=object;
        object.name="DeepSeek - positioned cutout";
        if (p.angle) object.rotate(p.angle,AnchorPosition.MIDDLECENTER);
        var b=object.bounds;
        var fraction=p.width*crop[2]*100/(b[2].as("px")-b[0].as("px"));
        object.resize(fraction,fraction,AnchorPosition.MIDDLECENTER);
        b=object.bounds;
        var correction=p.width*crop[2]*100/(b[2].as("px")-b[0].as("px"));
        if (Math.abs(correction-100)>1) {
            object.resize(correction,correction,AnchorPosition.MIDDLECENTER);
            b=object.bounds;
        }
        object.translate(crop[0]+p.center[0]*crop[2]-(b[0].as("px")+b[2].as("px"))/2,
                         crop[1]+p.center[1]*crop[3]-(b[1].as("px")+b[3].as("px"))/2);
        b=object.bounds;
        if (b[0].as("px")<crop[0] || b[2].as("px")>crop[0]+crop[2] ||
            b[1].as("px")<crop[1] || b[3].as("px")>crop[1]+crop[3]) throw Error("Transformed object leaves crop");
        if (p.exposure) object.adjustLevels(0,255,Math.pow(2,p.exposure/70),0,255);
        var shadow=object.duplicate(object,ElementPlacement.PLACEAFTER);
        shadow.name="Contact shadow from cutout";
        doc.activeLayer=shadow;
        shadow.adjustLevels(0,255,1,0,1);
        shadow.translate(3,5);
        shadow.applyGaussianBlur(Math.max(3,p.width*crop[2]*.016));
        shadow.blendMode=BlendMode.MULTIPLY;
        shadow.opacity=p.shadow_opacity;
        var top=object;
        for (var i=0;i<p.occluders.length;i++) {
            var polygon=p.occluders[i], pts=[];
            for(var j=0;j<polygon.length;j++){
                pts.push([crop[0]+polygon[j][0]*crop[2],crop[1]+polygon[j][1]*crop[3]]);
            }
            var layer=base.duplicate(top,ElementPlacement.PLACEBEFORE);
            layer.name="DeepSeek - foreground occluder "+(i+1);
            doc.activeLayer=layer;
            doc.selection.select(pts,SelectionType.REPLACE,1.8,true);
            var c=charIDToTypeID, action=new ActionDescriptor(),ref=new ActionReference();
            action.putClass(c("Nw  "),c("Chnl"));
            ref.putEnumerated(c("Chnl"),c("Chnl"),c("Msk "));
            action.putReference(c("At  "),ref);
            action.putEnumerated(c("Usng"),c("UsrM"),c("RvlS"));
            executeAction(c("Mk  "),action,DialogModes.NO);
            doc.selection.deselect();
            top=layer;
        }
        doc.activeLayer=object;
        var options=new PhotoshopSaveOptions();
        options.layers=true;options.embedColorProfile=true;
        doc.saveAs(new File(job.psd),options,true);
        doc.saveAs(new File(job.png),new PNGSaveOptions(),true);
        return [canvasWidth,canvasHeight,b[0].as("px"),b[1].as("px"),b[2].as("px"),b[3].as("px"),p.occluders.length].join(",");
    } finally {
        if(cutout)cutout.close(SaveOptions.DONOTSAVECHANGES);
        if(doc)doc.close(SaveOptions.DONOTSAVECHANGES);
        if(prior)app.activeDocument=prior;
        app.preferences.rulerUnits=oldUnits;
        app.displayDialogs=dialogs;
    }
}
placeAsset(JOB);