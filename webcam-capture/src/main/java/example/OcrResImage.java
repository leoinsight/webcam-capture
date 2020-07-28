package example;

import java.util.ArrayList;

public class OcrResImage {
    private String uid;
    private String name;
    private String inferResult;
    private String message;
    private ArrayList<OcrResField> fields = new ArrayList<OcrResField>();
    private ValidationResult validationResult;

    // Getter Methods

    public String getUid() {
        return uid;
    }

    public String getName() {
        return name;
    }

    public String getInferResult() {
        return inferResult;
    }

    public String getMessage() {
        return message;
    }

    public ArrayList<OcrResField> getFields() {
        return fields;
    }

    public ValidationResult getValidationResult() {
        return validationResult;
    }

    // Setter Methods

    public void setUid(String uid) {
        this.uid = uid;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setInferResult(String inferResult) {
        this.inferResult = inferResult;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public void setFields(ArrayList<OcrResField> fields) {
        this.fields = fields;
    }

    public void setValidationResult(ValidationResult validationResult) {
        this.validationResult = validationResult;
    }

}

class ValidationResult {
    private String result;

    // Getter Methods

    public String getResult() {
        return result;
    }

    // Setter Methods

    public void setResult(String result) {
        this.result = result;
    }
}

