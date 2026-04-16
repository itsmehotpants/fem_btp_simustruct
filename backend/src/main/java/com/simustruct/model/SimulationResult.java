package com.simustruct.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public class SimulationResult {
    private List<List<Double>> nodeCoords;
    private List<Double> stressVm;
    private List<Double> displacementX;
    private List<Double> displacementY;
    private double scf;
    private double sigmaMaxMpa;
    private double safetyFactor;
    private double inferenceMs;
    private int nNodes;
    private String modelType;
    private String warning;

    // Error fields (only when FEM validated)
    private List<Double> femStressVm;
    private Double femTimeMs;
    private Double errorPct;

    // Static factory for error responses
    public static SimulationResult error(String message) {
        SimulationResult r = new SimulationResult();
        r.setWarning(message);
        return r;
    }

    // Getters and Setters
    public List<List<Double>> getNodeCoords() { return nodeCoords; }
    public void setNodeCoords(List<List<Double>> nodeCoords) { this.nodeCoords = nodeCoords; }
    public List<Double> getStressVm() { return stressVm; }
    public void setStressVm(List<Double> stressVm) { this.stressVm = stressVm; }
    public List<Double> getDisplacementX() { return displacementX; }
    public void setDisplacementX(List<Double> displacementX) { this.displacementX = displacementX; }
    public List<Double> getDisplacementY() { return displacementY; }
    public void setDisplacementY(List<Double> displacementY) { this.displacementY = displacementY; }
    public double getScf() { return scf; }
    public void setScf(double scf) { this.scf = scf; }
    public double getSigmaMaxMpa() { return sigmaMaxMpa; }
    public void setSigmaMaxMpa(double sigmaMaxMpa) { this.sigmaMaxMpa = sigmaMaxMpa; }
    public double getSafetyFactor() { return safetyFactor; }
    public void setSafetyFactor(double safetyFactor) { this.safetyFactor = safetyFactor; }
    public double getInferenceMs() { return inferenceMs; }
    public void setInferenceMs(double inferenceMs) { this.inferenceMs = inferenceMs; }
    public int getNNodes() { return nNodes; }
    public void setNNodes(int nNodes) { this.nNodes = nNodes; }
    public String getModelType() { return modelType; }
    public void setModelType(String modelType) { this.modelType = modelType; }
    public String getWarning() { return warning; }
    public void setWarning(String warning) { this.warning = warning; }
    public List<Double> getFemStressVm() { return femStressVm; }
    public void setFemStressVm(List<Double> femStressVm) { this.femStressVm = femStressVm; }
    public Double getFemTimeMs() { return femTimeMs; }
    public void setFemTimeMs(Double femTimeMs) { this.femTimeMs = femTimeMs; }
    public Double getErrorPct() { return errorPct; }
    public void setErrorPct(Double errorPct) { this.errorPct = errorPct; }
}
