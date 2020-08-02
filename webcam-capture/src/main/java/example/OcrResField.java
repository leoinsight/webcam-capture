package example;

import java.util.ArrayList;

public class OcrResField {
    private String valueType;
    private String inferText;
    private float inferConfidence;
    private String type;
    private BoundingPoly boundingPoly;

    // Getter Methods

    public String getValueType() {
        return valueType;
    }

    public String getInferText() {
        return inferText;
    }

    public float getInferConfidence() {
        return inferConfidence;
    }

    public String getType() {
        return type;
    }

    public BoundingPoly getBoundingPoly() {
        return boundingPoly;
    }

    // Setter Methods

    public void setValueType(String valueType) {
        this.valueType = valueType;
    }

    public void setInferText(String inferText) {
        this.inferText = inferText;
    }

    public void setInferConfidence(float inferConfidence) {
        this.inferConfidence = inferConfidence;
    }

    public void setType(String type) {
        this.type = type;
    }

    public void setBoundingPoly(BoundingPoly boundingPoly) {
        this.boundingPoly = boundingPoly;
    }
}

class BoundingPoly {
    private ArrayList<Vertex> vertices = new ArrayList<Vertex>();

    public ArrayList<Vertex> getVertices() {
        return vertices;
    }

    public void setVertices(ArrayList<Vertex> vertices) {
        this.vertices = vertices;
    }
}

class Vertex {
    private float x;
    private float y;

    public float getX() {
        return x;
    }
    public void setX(float x) {
        this.x = x;
    }
    public float getY() {
        return y;
    }
    public void setY(float y) {
        this.y = y;
    }
}

