package example;

import java.util.ArrayList;

public class OcrRequest {
    private String version;
    private String requestId;
    private long timestamp;
    private String lang;
    private ArrayList<OcrReqImage> images = new ArrayList<OcrReqImage>();

    public OcrRequest(String version, String requestId, long timestamp, String lang) {
        this.version = version;
        this.requestId = requestId;
        this.timestamp = timestamp;
        this.lang = lang;
    }

    // Getter Methods

    public String getVersion() {
        return version;
    }

    public String getRequestId() {
        return requestId;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public String getLang() {
        return lang;
    }

    public ArrayList<OcrReqImage> getImages() {
        return images;
    }

    // Setter Methods

    public void setVersion(String version) {
        this.version = version;
    }

    public void setRequestId(String requestId) {
        this.requestId = requestId;
    }

    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }

    public void setLang(String lang) {
        this.lang = lang;
    }

    public void setImages(ArrayList<OcrReqImage> images) {
        this.images = images;
    }
}
