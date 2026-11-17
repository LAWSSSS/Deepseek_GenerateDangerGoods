/* Host-owned readback. ActionManager values are observed, never model estimates. */
function psKey(id) {
    try { var s = typeIDToStringID(id); if (s) return s; } catch (e) {}
    try { return typeIDToCharID(id); } catch (e2) { return String(id); }
}
function psValue(container, key, type) {
    if (type === DescValueType.BOOLEANTYPE) return container.getBoolean(key);
    if (type === DescValueType.INTEGERTYPE) return container.getInteger(key);
    if (type === DescValueType.DOUBLETYPE) return container.getDouble(key);
    if (type === DescValueType.STRINGTYPE) return container.getString(key);
    if (type === DescValueType.UNITDOUBLE) return {unit:psKey(container.getUnitDoubleType(key)),value:container.getUnitDoubleValue(key)};
    if (type === DescValueType.ENUMERATEDTYPE) return psKey(container.getEnumerationValue(key));
    if (type === DescValueType.OBJECTTYPE) return psDescriptor(container.getObjectValue(key));
    if (type === DescValueType.LISTTYPE) {
        var list = container.getList(key), values = [];
        for (var i=0; i<list.count; i++) values.push(psValue(list, i, list.getType(i)));
        return values;
    }
    if (type === DescValueType.REFERENCETYPE) {
        var ref = container.getReference(key);
        try { return {enumerated:psKey(ref.getEnumeratedValue())}; } catch (e) {}
        return {status:"unsupported_reference"};
    }
    return {status:"unsupported_descriptor_type"};
}
function psDescriptor(desc) {
    var value = {};
    for (var i=0; i<desc.count; i++) {
        var key = desc.getKey(i);
        value[psKey(key)] = psValue(desc, key, desc.getType(key));
    }
    return value;
}
function psDocumentState(doc) {
    app.activeDocument = doc;
    var layers = [], ids = [], s = stringIDToTypeID;
    for (var i=0; i<doc.layers.length; i++) {
        var layer = doc.layers[i], ref = new ActionReference();
        ref.putIdentifier(s("layer"), layer.id);
        var descriptor = executeActionGet(ref), clipped = false;
        try { clipped = layer.grouped; } catch (e) {}
        var item = {id:layer.id,name:layer.name,visible:layer.visible,clipped:clipped,
            opacity:layer.opacity,blend_mode:String(layer.blendMode),kind:String(layer.kind)};
        var keys = ["adjustment","hasUserMask","userMaskEnabled","userMaskDensity",
                    "userMaskFeather","hasVectorMask","vectorMaskEnabled","layerSection"];
        for (var j=0; j<keys.length; j++) {
            var key = s(keys[j]);
            if (descriptor.hasKey(key)) item[keys[j]] = psValue(descriptor,key,descriptor.getType(key));
        }
        layers.push(item); ids.push(layer.id);
    }
    var document = {id:doc.id,name:doc.name,canvas:[doc.width.as("px"),doc.height.as("px")],
        mode:String(doc.mode),bits_per_channel:String(doc.bitsPerChannel),color_profile:doc.colorProfileName,
        history_name:doc.activeHistoryState.name,history_count:doc.historyStates.length};
    return {version:"photoshop-state-v2",document:document,layers:layers,
        id_scope:"document ID is session-local; layer IDs are read from the reopened PSD",
        basis:"Photoshop ActionManager and DOM readback; no inferred physical exposure or illumination"};
}
