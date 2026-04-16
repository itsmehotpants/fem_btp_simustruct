"""
SimuStruct AI V3 — FastAPI Compute Engine
==========================================
REST API serving AI inference and optional FEM validation.

Endpoints:
    POST /infer   — AI surrogate forward pass
    POST /fem     — FEniCSx ground-truth solver
    GET  /health  — Service health check
    GET  /materials — List available materials
"""

import os
import sys
import time
import numpy as np
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.materials import MATERIALS, CATEGORIES, get_material_summary
from src.inference import load_model, run_ai_inference
from src.fem_solver import run_fem


# ==================== APP SETUP ====================

app = FastAPI(
    title="SimuStruct AI Compute Engine",
    description="Real-time structural stress prediction via Deep Learning surrogates",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
MODEL = None


@app.on_event("startup")
async def startup_load_model():
    global MODEL
    try:
        MODEL = load_model()
        print("[Compute Engine] Model loaded successfully")
    except Exception as e:
        print(f"[Compute Engine] Model load failed: {e}")
        print("[Compute Engine] Will use analytical solver as fallback")


# ==================== REQUEST/RESPONSE MODELS ====================

class HoleGeometry(BaseModel):
    rx: float = 0.05
    ry: float = 0.05
    cx: float = 0.5
    cy: float = 0.5


class GeometryParams(BaseModel):
    plate_width: float = 1.0
    plate_height: float = 0.5
    holes: List[HoleGeometry] = [HoleGeometry()]
    mesh_size: float = 0.02


class LoadParams(BaseModel):
    type: str = "tension"
    magnitude: float = 1e5
    angle: float = 0.0


class SimulationRequest(BaseModel):
    geometry: GeometryParams = GeometryParams()
    material_key: str = "structural_steel_a36"
    load: LoadParams = LoadParams()
    run_fem_validation: bool = False
    n_grid: int = 80

    @field_validator("material_key")
    @classmethod
    def valid_material(cls, v):
        if v not in MATERIALS:
            raise ValueError(f"Unknown material: {v}. Available: {list(MATERIALS.keys())}")
        return v


class SimulationResponse(BaseModel):
    node_coords: List[List[float]]
    stress_vm: List[float]
    displacement_x: List[float]
    displacement_y: List[float]
    scf: float
    sigma_max_mpa: float
    safety_factor: float
    inference_ms: float
    n_nodes: int
    model_type: str = "enhanced_mlp"
    fem_stress_vm: Optional[List[float]] = None
    fem_time_ms: Optional[float] = None
    error_pct: Optional[float] = None


# ==================== ENDPOINTS ====================

@app.post("/infer", response_model=SimulationResponse)
async def infer(req: SimulationRequest):
    """
    Run AI surrogate inference for stress field prediction.
    Optionally validates against FEM ground truth.
    """
    mat = MATERIALS[req.material_key]

    # Convert geometry to dict format
    geometry = {
        "plate_width": req.geometry.plate_width,
        "plate_height": req.geometry.plate_height,
        "holes": [h.model_dump() for h in req.geometry.holes],
        "mesh_size": req.geometry.mesh_size,
    }
    load = req.load.model_dump()

    # Run AI inference
    try:
        result = run_ai_inference(
            geometry, mat, load,
            material_key=req.material_key,
            n_grid=req.n_grid,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # Optional FEM validation
    if req.run_fem_validation:
        try:
            fem_result = run_fem(
                geometry, mat, load,
                material_key=req.material_key,
            )
            result["fem_stress_vm"] = fem_result["stress_vm"].tolist()
            result["fem_time_ms"] = fem_result.get("fem_time_ms", 0)

            # Compute error
            ai_stress = np.array(result["stress_vm"])
            fem_stress = np.array(result["fem_stress_vm"])
            min_len = min(len(ai_stress), len(fem_stress))
            if min_len > 0:
                error = np.abs(ai_stress[:min_len] - fem_stress[:min_len])
                fem_mean = np.abs(fem_stress[:min_len]).mean()
                result["error_pct"] = float(error.mean() / (fem_mean + 1e-8) * 100)
        except Exception as e:
            result["fem_stress_vm"] = None
            result["fem_time_ms"] = None
            result["error_pct"] = None
            print(f"[FEM Validation] Failed: {e}")

    return SimulationResponse(**result)


@app.post("/fem")
async def fem_solve(req: SimulationRequest):
    """Run FEniCSx / analytical FEM solver directly."""
    mat = MATERIALS[req.material_key]
    geometry = {
        "plate_width": req.geometry.plate_width,
        "plate_height": req.geometry.plate_height,
        "holes": [h.model_dump() for h in req.geometry.holes],
        "mesh_size": req.geometry.mesh_size,
    }
    load = req.load.model_dump()

    try:
        result = run_fem(geometry, mat, load, material_key=req.material_key)
        # Convert numpy arrays to lists
        for key in result:
            if isinstance(result[key], np.ndarray):
                result[key] = result[key].tolist()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FEM solver failed: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "materials_count": len(MATERIALS),
        "version": "3.0.0",
    }


@app.get("/materials")
async def list_materials():
    """List all available materials grouped by category."""
    result = {}
    for cat in CATEGORIES:
        result[cat] = [
            {
                "key": k,
                "display_name": v["display_name"],
                "E_gpa": v["E"] / 1e9,
                "nu": v["nu"],
                "sigma_y_mpa": v["sigma_y"] / 1e6,
            }
            for k, v in MATERIALS.items()
            if v["category"] == cat
        ]
    return result


@app.get("/materials/{material_key}")
async def get_material(material_key: str):
    """Get detailed properties for a specific material."""
    if material_key not in MATERIALS:
        raise HTTPException(status_code=404, detail=f"Material not found: {material_key}")
    return {
        "key": material_key,
        **MATERIALS[material_key],
        "summary": get_material_summary(material_key),
    }


# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
