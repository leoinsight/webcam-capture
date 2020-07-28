package example;

import java.util.ArrayList;

public class OcrResponse {
    private String version;
    private String requestId;
    private float timestamp;
    private ArrayList<OcrResImage> images = new ArrayList<OcrResImage> ();

    // Getter Methods

    public String getVersion() {
        return version;
    }

    public String getRequestId() {
        return requestId;
    }

    public float getTimestamp() {
        return timestamp;
    }

    public ArrayList<OcrResImage> getImages() {
        return images;
    }

    // Setter Methods

    public void setVersion(String version) {
        this.version = version;
    }

    public void setRequestId(String requestId) {
        this.requestId = requestId;
    }

    public void setTimestamp(float timestamp) {
        this.timestamp = timestamp;
    }

    public void setImages(ArrayList<OcrResImage> images) {
        this.images = images;
    }

}

