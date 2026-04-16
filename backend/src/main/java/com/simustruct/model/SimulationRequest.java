package com.simustruct.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public class SimulationRequest {

    @Valid @NotNull
    private GeometryParams geometry;

    @NotBlank
    private String materialKey;

    @Valid @NotNull
    private LoadParams load;

    private boolean runFemValidation = false;

    // Getters and Setters
    public GeometryParams getGeometry() { return geometry; }
    public void setGeometry(GeometryParams geometry) { this.geometry = geometry; }
    public String getMaterialKey() { return materialKey; }
    public void setMaterialKey(String materialKey) { this.materialKey = materialKey; }
    public LoadParams getLoad() { return load; }
    public void setLoad(LoadParams load) { this.load = load; }
    public boolean isRunFemValidation() { return runFemValidation; }
    public void setRunFemValidation(boolean runFemValidation) { this.runFemValidation = runFemValidation; }

    // Inner classes
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class GeometryParams {
        private double plateWidth = 1.0;
        private double plateHeight = 0.5;
        private List<HoleParams> holes;
        private double meshSize = 0.02;

        public double getPlateWidth() { return plateWidth; }
        public void setPlateWidth(double plateWidth) { this.plateWidth = plateWidth; }
        public double getPlateHeight() { return plateHeight; }
        public void setPlateHeight(double plateHeight) { this.plateHeight = plateHeight; }
        public List<HoleParams> getHoles() { return holes; }
        public void setHoles(List<HoleParams> holes) { this.holes = holes; }
        public double getMeshSize() { return meshSize; }
        public void setMeshSize(double meshSize) { this.meshSize = meshSize; }
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class HoleParams {
        private double rx = 0.05;
        private double ry = 0.05;
        private double cx = 0.5;
        private double cy = 0.5;

        public double getRx() { return rx; }
        public void setRx(double rx) { this.rx = rx; }
        public double getRy() { return ry; }
        public void setRy(double ry) { this.ry = ry; }
        public double getCx() { return cx; }
        public void setCx(double cx) { this.cx = cx; }
        public double getCy() { return cy; }
        public void setCy(double cy) { this.cy = cy; }
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class LoadParams {
        private String type = "tension";
        private double magnitude = 1e5;
        private double angle = 0.0;

        public String getType() { return type; }
        public void setType(String type) { this.type = type; }
        public double getMagnitude() { return magnitude; }
        public void setMagnitude(double magnitude) { this.magnitude = magnitude; }
        public double getAngle() { return angle; }
        public void setAngle(double angle) { this.angle = angle; }
    }
}
