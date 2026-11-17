/* Fixed, host-owned exporter. JOB contains validated data, never executable text. */
function exportObject(job) {
    var prior = app.documents.length ? app.activeDocument : null;
    var dialogs = app.displayDialogs, units = app.preferences.rulerUnits;
    var source = null, doc = null, isolated = null, reopened = null;
    function png(document, path) {
        app.activeDocument = document;
        document.saveAs(new File(path), new PNGSaveOptions(), true);
    }
    function imported(path, name) {
        var opened = app.open(new File(path));
        try {
            if (opened.width.as("px") !== job.canvas[0] || opened.height.as("px") !== job.canvas[1])
                throw Error("photoshop_import_canvas_mismatch");
            var layer = opened.activeLayer.duplicate(doc, ElementPlacement.PLACEATBEGINNING);
            app.activeDocument = doc;
            layer.name = name;
            return layer;
        } finally { opened.close(SaveOptions.DONOTSAVECHANGES); app.activeDocument = doc; }
    }
    function curve(list, channel, delta, shape) {
        var c = charIDToTypeID, d = new ActionDescriptor(), ref = new ActionReference(), points = new ActionList();
        ref.putEnumerated(c("Chnl"), c("Chnl"), c(channel)); d.putReference(c("Chnl"), ref);
        var samples = [[0,0],[48,48+delta*.48],[128,128+delta],[208,208+delta*.48],[255,255]];
        if (shape === "black_point") samples = [[0,0],[48,48+delta],[128,128],[208,208],[255,255]];
        if (shape === "highlights") samples = [[0,0],[48,48],[128,128],[208,208+delta],[255,255]];
        for (var i=0; i<samples.length; i++) {
            var point = new ActionDescriptor();
            point.putDouble(c("Hrzn"), samples[i][0]); point.putDouble(c("Vrtc"), samples[i][1]);
            points.putObject(c("Pnt "), point);
        }
        d.putList(c("Crv "), points); list.putObject(c("CrvA"), d);
    }
    function adjustment(name, value) {
        var c = charIDToTypeID, settings = new ActionDescriptor(), type = "Crvs";
        if (name === "saturation") {
            var changes = new ActionList(), master = new ActionDescriptor();
            master.putInteger(c("H   "), 0); master.putInteger(c("Strt"), Math.round(value)); master.putInteger(c("Lght"), 0);
            changes.putObject(c("Hst2"), master); settings.putList(c("Adjs"), changes); type = "HStr";
        } else {
            var channels = new ActionList();
            if (name === "brightness") curve(channels, "Cmps", value);
            else if (name === "warmth") { curve(channels, "Rd  ", value); curve(channels, "Bl  ", -value); }
            else if (name === "black_point" || name === "highlights") curve(channels, "Cmps", value, name);
            else if (name === "tint") { curve(channels, "Grn ", -value); }
            else throw Error("photoshop_unknown_adjustment");
            settings.putList(c("Adjs"), channels);
        }
        var action = new ActionDescriptor(), reference = new ActionReference(), layer = new ActionDescriptor();
        reference.putClass(c("AdjL")); action.putReference(c("null"), reference);
        layer.putString(c("Nm  "), "Object adjustment | " + name); layer.putObject(c("Type"), c(type), settings);
        action.putObject(c("Usng"), c("AdjL"), layer); executeAction(c("Mk  "), action, DialogModes.NO);
        doc.activeLayer.grouped = true;
        if (job.action_masks && job.action_masks[name]) {
            var adjustmentLayer = doc.activeLayer;
            var maskLayer = imported(job.action_masks[name], "Temporary registered part mask");
            doc.activeLayer = maskLayer;
            var selection = new ActionDescriptor(), selectionRef = new ActionReference(), alphaRef = new ActionReference();
            selectionRef.putProperty(c("Chnl"),c("fsel")); selection.putReference(c("null"),selectionRef);
            alphaRef.putEnumerated(c("Chnl"),c("Chnl"),c("Trsp")); selection.putReference(c("T   "),alphaRef);
            executeAction(c("setd"),selection,DialogModes.NO);
            maskLayer.remove(); doc.activeLayer = adjustmentLayer;
            // Replace the initial white adjustment mask with the registered selection.
            var remove = new ActionDescriptor(), oldMask = new ActionReference();
            oldMask.putEnumerated(c("Chnl"),c("Chnl"),c("Msk ")); remove.putReference(c("null"),oldMask);
            executeAction(c("Dlt "),remove,DialogModes.NO);
            var makeMask = new ActionDescriptor(), at = new ActionReference();
            makeMask.putClass(c("Nw  "),c("Chnl")); at.putEnumerated(c("Chnl"),c("Chnl"),c("Msk "));
            makeMask.putReference(c("At  "),at); makeMask.putEnumerated(c("Usng"),c("UsrM"),c("RvlS"));
            executeAction(c("Mk  "),makeMask,DialogModes.NO); doc.selection.deselect();
        }
    }
    try {
        app.displayDialogs = DialogModes.NO; app.preferences.rulerUnits = Units.PIXELS;
        source = app.open(new File(job.background));
        doc = source.duplicate("Scrap isolated object export", false); app.activeDocument = doc;
        if (doc.mode !== DocumentMode.RGB || doc.bitsPerChannel !== BitsPerChannelType.EIGHT)
            throw Error("photoshop_expected_8bit_RGB");
        if (doc.width.as("px") !== job.canvas[0] || doc.height.as("px") !== job.canvas[1])
            throw Error("photoshop_background_canvas_mismatch");
        doc.layers[0].name = "Background | source pixels";
        if (job.shadow_rgba) imported(job.shadow_rgba, "Shadow | verified RGBA");
        var object = imported(job.object_rgba, "Object | " + job.object_id);
        var foreground = job.foreground_rgba ? imported(job.foreground_rgba, "Foreground and protected | original pixels") : null;
        png(doc, job.outputs.baseline);
        doc.activeLayer = object;
        var applied = 0, names = ["brightness", "black_point", "highlights", "warmth", "tint", "saturation"];
        for (var i=0; i<names.length; i++) {
            if (job.actions[names[i]] !== undefined) { adjustment(names[i], job.actions[names[i]]); applied++; }
        }
        // Clipping confines every adjustment to the source object alpha. The
        // foreground stays above the adjustment chain and is never filtered.
        png(doc, job.outputs.png);
        var options = new PhotoshopSaveOptions(); options.layers = true; options.embedColorProfile = true;
        doc.saveAs(new File(job.outputs.psd), options, true);
        isolated = doc.duplicate("Object alpha verification", false); app.activeDocument = isolated;
        for (var j=0; j<isolated.layers.length; j++) {
            var label = isolated.layers[j].name;
            isolated.layers[j].visible = label === "Object | " + job.object_id || label.indexOf("Object adjustment | ") === 0;
        }
        png(isolated, job.outputs.object_alpha);
        isolated.close(SaveOptions.DONOTSAVECHANGES); isolated = null;
        doc.close(SaveOptions.DONOTSAVECHANGES); doc = null;
        reopened = app.open(new File(job.outputs.psd)); png(reopened, job.outputs.reopened);
        var actualState = psDocumentState(reopened), layers = actualState.layers;
        return JSON.stringify({canvas:[reopened.width.as("px"), reopened.height.as("px")],
            applied:applied, layers:layers, color_profile:reopened.colorProfileName,
            mode:"RGB8", reopened:true, state:actualState});
    } finally {
        if (reopened) reopened.close(SaveOptions.DONOTSAVECHANGES);
        if (isolated) isolated.close(SaveOptions.DONOTSAVECHANGES);
        if (doc) doc.close(SaveOptions.DONOTSAVECHANGES);
        if (source) source.close(SaveOptions.DONOTSAVECHANGES);
        if (prior) app.activeDocument = prior;
        app.displayDialogs = dialogs; app.preferences.rulerUnits = units;
    }
}
exportObject(JOB);
