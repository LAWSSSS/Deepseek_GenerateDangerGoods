/* Trusted Photoshop operations. The host supplies validated JSON data. */
function runEdit(job) {
    var previous = app.documents.length ? app.activeDocument : null;
    var oldDialogs = app.displayDialogs;
    var oldUnits = app.preferences.rulerUnits;
    var opened = null, doc = null, preview = null;
    var file = new File(job.source);
    try {
        app.displayDialogs = DialogModes.NO;
        app.preferences.rulerUnits = Units.PIXELS;
        var source = null;
        for (var i = 0; i < app.documents.length; i++) {
            try {
                if (app.documents[i].fullName.fsName.toLowerCase() === file.fsName.toLowerCase()) {
                    source = app.documents[i];
                    break;
                }
            } catch (unsaved) {}
        }
        if (source && !source.saved) throw Error("Save the source Photoshop tab before running");
        if (!source) source = opened = app.open(file);
        app.activeDocument = source;
        doc = source.duplicate("PS model edit", false);
        app.activeDocument = doc;
        var width = doc.width.as("px"), height = doc.height.as("px");
        if (doc.mode !== DocumentMode.RGB || doc.bitsPerChannel !== BitsPerChannelType.EIGHT) {
            throw Error("Expected 8-bit RGB document");
        }
        if (job.group) {
            for (var j = 0; j < job.hide_groups.length; j++) {
                doc.layerSets.getByName(job.hide_groups[j]).visible = false;
            }
            doc.layerSets.getByName(job.group).visible = true;
        }
        var targetGroup = job.group ? doc.layerSets.getByName(job.group) : null;
        for (var n = 0; n < job.ops.length; n++) {
            var op = job.ops[n], region = job.targets[op.target];
            var top = doc.layers[0];
            var layer = applyOperation(doc, targetGroup, op, region, width, height);
            if (targetGroup && layer.parent !== targetGroup) layer.move(targetGroup, ElementPlacement.INSIDE);
            if (!targetGroup && layer.parent !== doc) layer.move(top, ElementPlacement.PLACEBEFORE);
        }
        if (job.psd) {
            var options = new PhotoshopSaveOptions();
            options.layers = true;
            options.embedColorProfile = true;
            doc.saveAs(new File(job.psd), options, true);
        }
        if (job.png) doc.saveAs(new File(job.png), new PNGSaveOptions(), true);
        if (job.preview) {
            preview = doc.duplicate("PS model preview", true);
            app.activeDocument = preview;
            var rect = job.preview_rect;
            preview.crop([UnitValue(rect[0]*width,"px"),UnitValue(rect[1]*height,"px"),
                          UnitValue(rect[2]*width,"px"),UnitValue(rect[3]*height,"px")]);
            var w = preview.width.as("px"), h = preview.height.as("px");
            var scale = Math.min(1, job.preview_edge / Math.max(w,h));
            if (scale < 1) {
                preview.resizeImage(UnitValue(Math.max(1, Math.round(w*scale)),"px"),
                                    UnitValue(Math.max(1, Math.round(h*scale)),"px"),
                                    null,ResampleMethod.BICUBIC);
            }
            var jpeg = new JPEGSaveOptions();
            jpeg.quality = 6;
            jpeg.embedColorProfile = true;
            preview.saveAs(new File(job.preview),jpeg,true,Extension.LOWERCASE);
        }
        return JSON.stringify({width:width,height:height,applied:job.ops.length});
    } finally {
        if (preview) preview.close(SaveOptions.DONOTSAVECHANGES);
        if (doc) doc.close(SaveOptions.DONOTSAVECHANGES);
        if (opened) opened.close(SaveOptions.DONOTSAVECHANGES);
        if (previous) app.activeDocument = previous;
        app.displayDialogs = oldDialogs;
        app.preferences.rulerUnits = oldUnits;
    }
}
function selectTarget(doc, region, width, height) {
    var pts = [];
    for (var i=0;i<region.polygon.length;i++) {
        pts.push([region.polygon[i][0]*width,region.polygon[i][1]*height]);
    }
    doc.selection.select(pts,SelectionType.REPLACE,region.feather_px,true);
}
function addMask() {
    var c=charIDToTypeID, action=new ActionDescriptor(), ref=new ActionReference();
    action.putClass(c("Nw  "),c("Chnl"));
    ref.putEnumerated(c("Chnl"),c("Chnl"),c("Msk "));
    action.putReference(c("At  "),ref);
    action.putEnumerated(c("Usng"),c("UsrM"),c("RvlS"));
    executeAction(c("Mk  "),action,DialogModes.NO);
}
function makeAdjustment(name, type, settings) {
    var c=charIDToTypeID, action=new ActionDescriptor(), ref=new ActionReference();
    ref.putClass(c("AdjL"));
    action.putReference(c("null"),ref);
    var layer=new ActionDescriptor();
    layer.putString(c("Nm  "),name);
    layer.putObject(c("Type"),c(type),settings);
    action.putObject(c("Usng"),c("AdjL"),layer);
    executeAction(c("Mk  "),action,DialogModes.NO);
}
function applyOperation(doc, targetGroup, op, region, width, height) {
    var c=charIDToTypeID, settings=new ActionDescriptor();
    var name=op.target+" | "+op.op+" "+op.value;
    if (targetGroup) {
        doc.activeLayer=targetGroup.layers.length ? targetGroup.layers[0] : targetGroup;
    } else {
        for (var i=0; i<doc.layers.length; i++) {
            if (doc.layers[i].visible) { doc.activeLayer=doc.layers[i]; break; }
        }
    }
    if (op.op === "blur_px") {
        var merged=doc.duplicate("PS blur source",true);
        var raster;
        try { raster=merged.activeLayer.duplicate(doc,ElementPlacement.PLACEATBEGINNING); }
        finally { merged.close(SaveOptions.DONOTSAVECHANGES); }
        app.activeDocument=doc;
        doc.activeLayer=raster;
        raster.name=name;
        raster.applyGaussianBlur(op.value);
        selectTarget(doc,region,width,height);
        addMask();
    } else {
        selectTarget(doc,region,width,height);
        if (op.op === "saturation") {
            var changes=new ActionList(), master=new ActionDescriptor();
            master.putInteger(c("H   "),0);
            master.putInteger(c("Strt"),op.value);
            master.putInteger(c("Lght"),0);
            changes.putObject(c("Hst2"),master);
            settings.putList(c("Adjs"),changes);
            makeAdjustment(name,"HStr",settings);
        } else {
            var channels=new ActionList();
            if (op.op === "warmth") {
                curveChannel(channels,"Rd  ",op.value);
                curveChannel(channels,"Bl  ",-op.value);
            } else if (op.op === "brightness") {
                curveChannel(channels,"Cmps",op.value);
            } else {
                throw Error("Unexpected operation");
            }
            settings.putList(c("Adjs"),channels);
            makeAdjustment(name,"Crvs",settings);
        }
    }
    doc.selection.deselect();
    return doc.activeLayer;
}
function curveChannel(list, channel, delta) {
    var c=charIDToTypeID, curve=new ActionDescriptor(), ref=new ActionReference();
    ref.putEnumerated(c("Chnl"),c("Chnl"),c(channel));
    curve.putReference(c("Chnl"),ref);
    var samples=[[0,0],[48,48+delta*.48],[128,128+delta],[208,208+delta*.48],[255,255]];
    var points=new ActionList();
    for (var i=0;i<samples.length;i++) {
        var point=new ActionDescriptor();
        point.putDouble(c("Hrzn"),samples[i][0]);
        point.putDouble(c("Vrtc"),samples[i][1]);
        points.putObject(c("Pnt "),point);
    }
    curve.putList(c("Crv "),points);
    list.putObject(c("CrvA"),curve);
}
runEdit(JOB);