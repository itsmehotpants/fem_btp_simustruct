package com.simustruct.controller;

import com.simustruct.model.SimulationRequest;
import com.simustruct.model.SimulationResult;
import com.simustruct.model.ValidationResult;
import com.simustruct.service.ComputeEngineClient;
import com.simustruct.service.OODValidationService;

import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

/**
 * SimuStruct AI — Main REST Controller
 *
 * Handles simulation requests, validates inputs against training
 * distribution (OOD gatekeeper), and forwards to compute engine.
 */
@RestController
@RequestMapping("/api/v1")
@CrossOrigin(origins = {"http://localhost:8501", "http://localhost:3000", "*"})
public class SimulationController {

    @Autowired
    private ComputeEngineClient computeClient;

    @Autowired
    private OODValidationService oodService;

    /**
     * POST /api/v1/simulate
     * Run AI surrogate simulation with OOD validation.
     */
    @PostMapping("/simulate")
    public ResponseEntity<?> simulate(@RequestBody @Valid SimulationRequest request) {
        // OOD gatekeeper
        ValidationResult validation = oodService.validate(request);
        if (!validation.isValid()) {
            return ResponseEntity.badRequest().body(Map.of(
                "error", validation.getMessage(),
                "type", validation.getType()
            ));
        }

        // Forward to Python compute engine
        try {
            SimulationResult result = computeClient.callInfer(request);

            // Add validation warnings
            if (validation.hasWarning()) {
                result.setWarning(validation.getMessage());
            }

            return ResponseEntity.ok(result);
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body(Map.of(
                "error", "Compute engine error: " + e.getMessage()
            ));
        }
    }

    /**
     * POST /api/v1/validate
     * Validate inputs without running simulation.
     */
    @PostMapping("/validate")
    public ResponseEntity<ValidationResult> validateInput(
            @RequestBody @Valid SimulationRequest request) {
        ValidationResult result = oodService.validate(request);
        return ResponseEntity.ok(result);
    }

    /**
     * GET /api/v1/health
     * Health check endpoint.
     */
    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> health() {
        boolean computeHealthy = computeClient.isHealthy();
        return ResponseEntity.ok(Map.of(
            "status", computeHealthy ? "ok" : "degraded",
            "controller", "ok",
            "compute_engine", computeHealthy ? "ok" : "unreachable",
            "version", "3.0.0"
        ));
    }
}
