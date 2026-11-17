/* A registered PSD is opened read-only, exported for measurement, and closed unsaved. */
function inspectRegisteredPSD(job) {
    var prior = app.documents.length ? app.activeDocument : null, doc = null;
    var dialogs = app.displayDialogs, units = app.preferences.rulerUnits;
    try {
        app.displayDialogs = DialogModes.NO; app.preferences.rulerUnits = Units.PIXELS;
        doc = app.open(new File(job.psd));
        var state = psDocumentState(doc);
        doc.saveAs(new File(job.preview), new PNGSaveOptions(), true);
        return JSON.stringify(state);
    } finally {
        if (doc) doc.close(SaveOptions.DONOTSAVECHANGES);
        if (prior) app.activeDocument = prior;
        app.displayDialogs = dialogs; app.preferences.rulerUnits = units;
    }
}
inspectRegisteredPSD(JOB);
