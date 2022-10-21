
selectTMACores();
createAnnotationsFromPixelClassifier("Tissue thresholder", 100000.0, 10000.0, "SPLIT", "SELECT_NEW")
for (anno in getAnnotationObjects()){
    anno.setPathClass(getPathClass("Region*"))
}