package example;

public class OcrReqImage {
    private String format;
    private String name;
    private String url;
    private String data;

    public OcrReqImage(String format, String name) {
        this.format = format;
        this.name = name;
    }

    // Getter Methods

    public String getFormat() {
        return format;
    }

    public String getName() {
        return name;
    }

    public String getUrl() {
        return url;
    }

    public String getData() {
        return data;
    }

    // Setter Methods

    public void setFormat(String format) {
        this.format = format;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setUrl(String url) {
        this.url = url;
    }

    public void setData(String data) {
        this.data = data;
    }
}

