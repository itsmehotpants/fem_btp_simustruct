package com.simustruct.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.simustruct.model.SimulationRequest;
import com.simustruct.model.SimulationResult;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Map;

/**
 * HTTP client for the Python FastAPI compute engine.
 */
@Service
public class ComputeEngineClient {

    @Value("${compute.engine.url:http://localhost:8000}")
    private String computeEngineUrl;

    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;

    public ComputeEngineClient() {
        this.httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(10))
            .build();
        this.objectMapper = new ObjectMapper();
    }

    /**
     * Forward simulation request to FastAPI /infer endpoint.
     */
    public SimulationResult callInfer(SimulationRequest request) throws Exception {
        // Convert to FastAPI request format
        Map<String, Object> payload = Map.of(
            "geometry", Map.of(
                "plate_width", request.getGeometry().getPlateWidth(),
                "plate_height", request.getGeometry().getPlateHeight(),
                "holes", request.getGeometry().getHoles(),
                "mesh_size", request.getGeometry().getMeshSize()
            ),
            "material_key", request.getMaterialKey(),
            "load", Map.of(
                "type", request.getLoad().getType(),
                "magnitude", request.getLoad().getMagnitude(),
                "angle", request.getLoad().getAngle()
            ),
            "run_fem_validation", request.isRunFemValidation()
        );

        String jsonBody = objectMapper.writeValueAsString(payload);

        HttpRequest httpRequest = HttpRequest.newBuilder()
            .uri(URI.create(computeEngineUrl + "/infer"))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
            .timeout(Duration.ofSeconds(30))
            .build();

        HttpResponse<String> response = httpClient.send(
            httpRequest, HttpResponse.BodyHandlers.ofString()
        );

        if (response.statusCode() != 200) {
            throw new RuntimeException("Compute engine returned " + response.statusCode() +
                ": " + response.body());
        }

        return objectMapper.readValue(response.body(), SimulationResult.class);
    }

    /**
     * Check if the compute engine is reachable.
     */
    public boolean isHealthy() {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(computeEngineUrl + "/health"))
                .GET()
                .timeout(Duration.ofSeconds(5))
                .build();

            HttpResponse<String> response = httpClient.send(
                request, HttpResponse.BodyHandlers.ofString()
            );
            return response.statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }
}
