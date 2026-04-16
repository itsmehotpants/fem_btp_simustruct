package com.simustruct.service;

import com.simustruct.model.SimulationRequest;
import com.simustruct.model.ValidationResult;
import org.springframework.stereotype.Service;

/**
 * OOD (Out-of-Distribution) Validation Service
 *
 * Validates that simulation inputs fall within the training distribution
 * of the AI model. Rejects or warns on out-of-bounds geometries and loads.
 */
@Service
public class OODValidationService {

    // Training distribution bounds
    private static final double TRAINING_LOAD_MAX = 1e6;
    private static final double TRAINING_HOLE_RX_MAX = 0.15;
    private static final double TRAINING_HOLE_RY_MAX = 0.15;
    private static final double TRAINING_PLATE_W_MIN = 0.5;
    private static final double TRAINING_PLATE_W_MAX = 2.0;
    private static final double TRAINING_PLATE_H_MIN = 0.3;
    private static final double TRAINING_PLATE_H_MAX = 1.5;
    private static final double BOUNDARY_CLEARANCE = 0.05; // 5% clearance

    public ValidationResult validate(SimulationRequest req) {
        // 1. Plate dimensions check
        double W = req.getGeometry().getPlateWidth();
        double H = req.getGeometry().getPlateHeight();

        if (W < TRAINING_PLATE_W_MIN || W > TRAINING_PLATE_W_MAX) {
            return ValidationResult.fail(
                "Plate width " + W + "m is outside training range [" +
                TRAINING_PLATE_W_MIN + ", " + TRAINING_PLATE_W_MAX + "]"
            );
        }
        if (H < TRAINING_PLATE_H_MIN || H > TRAINING_PLATE_H_MAX) {
            return ValidationResult.fail(
                "Plate height " + H + "m is outside training range [" +
                TRAINING_PLATE_H_MIN + ", " + TRAINING_PLATE_H_MAX + "]"
            );
        }

        // 2. Hole geometry checks
        if (req.getGeometry().getHoles() != null) {
            for (int i = 0; i < req.getGeometry().getHoles().size(); i++) {
                var hole = req.getGeometry().getHoles().get(i);
                double rx = hole.getRx();
                double ry = hole.getRy();
                double cx = hole.getCx();
                double cy = hole.getCy();

                // Hole size check
                if (rx > TRAINING_HOLE_RX_MAX || ry > TRAINING_HOLE_RY_MAX) {
                    return ValidationResult.fail(
                        "Hole " + (i+1) + " radius exceeds training range (max " +
                        TRAINING_HOLE_RX_MAX + "m)"
                    );
                }

                // Boundary clearance check
                double clearanceX = Math.min(cx * W - rx, (1 - cx) * W - rx);
                double clearanceY = Math.min(cy * H - ry, (1 - cy) * H - ry);

                if (clearanceX < BOUNDARY_CLEARANCE * W) {
                    return ValidationResult.fail(
                        "Hole " + (i+1) + " too close to left/right boundary. " +
                        "Minimum clearance: " + (BOUNDARY_CLEARANCE * 100) + "% of width."
                    );
                }
                if (clearanceY < BOUNDARY_CLEARANCE * H) {
                    return ValidationResult.fail(
                        "Hole " + (i+1) + " too close to top/bottom boundary."
                    );
                }

                // Hole overlap check with other holes
                for (int j = i + 1; j < req.getGeometry().getHoles().size(); j++) {
                    var other = req.getGeometry().getHoles().get(j);
                    double dx = (cx - other.getCx()) * W;
                    double dy = (cy - other.getCy()) * H;
                    double dist = Math.sqrt(dx * dx + dy * dy);
                    double minDist = Math.max(rx, ry) + Math.max(other.getRx(), other.getRy());

                    if (dist < minDist * 1.2) {
                        return ValidationResult.fail(
                            "Holes " + (i+1) + " and " + (j+1) + " are overlapping or too close."
                        );
                    }
                }
            }
        }

        // 3. Load check
        double loadMag = req.getLoad().getMagnitude();
        if (loadMag > TRAINING_LOAD_MAX * 1.2) {
            return ValidationResult.warn(
                "Load magnitude " + loadMag + " Pa exceeds training range (" +
                TRAINING_LOAD_MAX + " Pa). Prediction accuracy may be reduced."
            );
        }

        if (loadMag <= 0) {
            return ValidationResult.fail("Load magnitude must be positive.");
        }

        return ValidationResult.pass();
    }
}
