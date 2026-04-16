package com.simustruct.model;

public class ValidationResult {
    private boolean valid;
    private String message;
    private String type; // "pass", "warn", "fail"

    private ValidationResult(boolean valid, String message, String type) {
        this.valid = valid;
        this.message = message;
        this.type = type;
    }

    public static ValidationResult pass() {
        return new ValidationResult(true, "Input validated successfully.", "pass");
    }

    public static ValidationResult warn(String message) {
        return new ValidationResult(true, message, "warn");
    }

    public static ValidationResult fail(String message) {
        return new ValidationResult(false, message, "fail");
    }

    public boolean isValid() { return valid; }
    public String getMessage() { return message; }
    public String getType() { return type; }
    public boolean hasWarning() { return "warn".equals(type); }
}
