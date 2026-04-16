#!/usr/bin/env python3
"""
SimuStruct AI V3 — Streamlit Frontend
=======================================
Professional engineering analysis platform with 4 tabs:
1. Design Studio — Material selection, geometry, load configuration
2. Simulation Hub — 3-panel AI/FEM/Error visualization
3. Model Analytics — Training metrics, R² scores, error distributions
4. Export Report — PDF generation & developer credits

Usage:
    streamlit run app.py
"""

import os
import sys
import json
import time
import numpy as np

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.materials import MATERIALS, CATEGORIES, get_material_features, get_shear_modulus, get_bulk_modulus
from src.fem_solver import analytical_stress_field
from src.scf_library import compare_scf, scf_elliptical_hole_infinite_plate
from src.fatigue import estimate_fatigue_life, generate_sn_curve

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="SimuStruct AI — Structural Analysis Platform",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load custom CSS
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ==================== CUSTOM COLORMAPS ====================
STRESS_CMAP = LinearSegmentedColormap.from_list(
    "simustruct_stress",
    ["#0d1b2a", "#1b3a4b", "#3a7ca5", "#4A90D9", "#f0c808", "#FF9F1C", "#FF6B6B", "#d00000"],
    N=256,
)

ERROR_CMAP = LinearSegmentedColormap.from_list(
    "simustruct_error",
    ["#0d1b2a", "#1b4332", "#3DD68C", "#f0c808", "#FF9F1C", "#FF6B6B", "#d00000"],
    N=256,
)


# ==================== HELPER FUNCTIONS ====================
def render_heatmap(coords, values, title, cmap=None, label="Von Mises (MPa)", holes=None, plate_w=1.0, plate_h=0.5):
    """Render a stress/displacement heatmap using matplotlib."""
    if cmap is None:
        cmap = STRESS_CMAP

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")

    coords = np.array(coords)
    values = np.array(values)

    n = int(np.sqrt(len(coords)))
    if n * n == len(coords):
        X = coords[:, 0].reshape(n, n)
        Y = coords[:, 1].reshape(n, n)
        Z = values.reshape(n, n)
        c = ax.pcolormesh(X, Y, Z, cmap=cmap, shading="gouraud")
    else:
        c = ax.tricontourf(coords[:, 0], coords[:, 1], values, levels=50, cmap=cmap)

    cbar = fig.colorbar(c, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label(label, color="#8892b0", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#8892b0")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8892b0", fontsize=8)

    # Draw holes
    if holes:
        for h in holes:
            cx = h["cx"] * plate_w
            cy = h["cy"] * plate_h
            ellipse = Ellipse(
                (cx, cy), 2 * h["rx"], 2 * h["ry"],
                fill=True, facecolor="#0f1117", edgecolor="#4A90D9",
                linewidth=1.5, linestyle="--", alpha=0.9,
            )
            ax.add_patch(ellipse)

    # Draw plate boundary
    ax.plot([0, plate_w, plate_w, 0, 0], [0, 0, plate_h, plate_h, 0],
            color="#4A90D9", linewidth=1.5, alpha=0.6)

    ax.set_xlabel("X (m)", color="#8892b0", fontsize=9)
    ax.set_ylabel("Y (m)", color="#8892b0", fontsize=9)
    ax.set_title(title, color="#e8eaf6", fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(colors="#8892b0", labelsize=8)
    ax.set_aspect("equal")

    for spine in ax.spines.values():
        spine.set_color("#2e3250")

    fig.tight_layout()
    return fig


def render_geometry_preview(W, H, holes, load_type, load_mag):
    """Render a geometry preview with holes and load arrows."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")

    # Plate
    plate = plt.Rectangle((0, 0), W, H, fill=True, facecolor="#21253a",
                           edgecolor="#4A90D9", linewidth=2)
    ax.add_patch(plate)

    # Holes
    for i, h in enumerate(holes):
        cx = h["cx"] * W
        cy = h["cy"] * H
        ellipse = Ellipse(
            (cx, cy), 2 * h["rx"], 2 * h["ry"],
            fill=True, facecolor="#0f1117", edgecolor="#FF9F1C",
            linewidth=2,
        )
        ax.add_patch(ellipse)
        ax.annotate(f"H{i+1}", (cx, cy), color="#FF9F1C",
                    fontsize=9, ha="center", va="center", fontweight="bold")

    # Load arrows
    arrow_color = "#3DD68C"
    n_arrows = 5
    if "tension" in load_type.lower() or "axial" in load_type.lower():
        for y in np.linspace(H * 0.1, H * 0.9, n_arrows):
            ax.annotate("", xy=(W + 0.05, y), xytext=(W + 0.15, y),
                        arrowprops=dict(arrowstyle="<-", color=arrow_color, lw=2))
        ax.text(W + 0.18, H / 2, f"{load_mag:.0f}\nkN/m²", color=arrow_color,
                fontsize=8, va="center", fontweight="bold")
    elif "shear" in load_type.lower():
        for x in np.linspace(W * 0.1, W * 0.9, n_arrows):
            ax.annotate("", xy=(x, H + 0.03), xytext=(x, H + 0.1),
                        arrowprops=dict(arrowstyle="<-", color=arrow_color, lw=2))

    # Fixed support (left edge)
    for y in np.linspace(0, H, 8):
        ax.plot([-0.03, 0], [y, y], color="#FF6B6B", linewidth=1.5)
    ax.plot([0, 0], [0, H], color="#FF6B6B", linewidth=3)
    ax.text(-0.08, H / 2, "Fixed", color="#FF6B6B", fontsize=8,
            rotation=90, va="center", ha="center")

    ax.set_xlim(-0.15, W + 0.25)
    ax.set_ylim(-0.1, H + 0.15)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", color="#8892b0", fontsize=9)
    ax.set_ylabel("Y (m)", color="#8892b0", fontsize=9)
    ax.set_title("Geometry & Loading Configuration", color="#e8eaf6",
                 fontsize=12, fontweight="bold")
    ax.tick_params(colors="#8892b0", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#2e3250")

    fig.tight_layout()
    return fig


# ==================== HERO BANNER ====================
st.markdown("""
<div class="hero-banner">
    <h1>⚙️ SimuStruct AI</h1>
    <p>Real-Time Structural Analysis via Deep Learning Surrogates</p>
    <p style="color: #4A90D9; font-size: 0.85rem; margin-top: 8px;">
        B.Tech Project • LNMIIT Jaipur • 22 Materials • Physics-Informed AI
    </p>
</div>
""", unsafe_allow_html=True)

# ==================== TABS ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "🎨  Design Studio",
    "🔬  Simulation Hub",
    "📊  Model Analytics",
    "📄  Export & Credits",
])

# ==================== TAB 1: DESIGN STUDIO ====================
with tab1:
    col_controls, col_preview = st.columns([1, 2])

    with col_controls:
        st.subheader("🧪 Material Selection")

        category = st.selectbox(
            "Category",
            CATEGORIES,
            index=0,
            key="mat_category",
        )

        mat_options = [k for k, v in MATERIALS.items() if v["category"] == category]
        material_key = st.selectbox(
            "Material",
            options=mat_options,
            format_func=lambda k: MATERIALS[k]["display_name"],
            key="material_key",
        )
        mat = MATERIALS[material_key]

        # Material property card
        G = get_shear_modulus(material_key) / 1e9
        K = get_bulk_modulus(material_key) / 1e9
        st.markdown(f"""
        <div class="material-card">
            <div class="prop-row"><span>Young's Modulus</span><b>{mat['E']/1e9:.1f} GPa</b></div>
            <div class="prop-row"><span>Poisson's Ratio</span><b>{mat['nu']:.3f}</b></div>
            <div class="prop-row"><span>Yield Strength</span><b>{mat['sigma_y']/1e6:.0f} MPa</b></div>
            <div class="prop-row"><span>UTS</span><b>{mat['UTS']/1e6:.0f} MPa</b></div>
            <div class="prop-row"><span>Density</span><b>{mat['rho']} kg/m³</b></div>
            <div class="prop-row"><span>Shear Modulus G</span><b>{G:.1f} GPa</b></div>
            <div class="prop-row"><span>Bulk Modulus K</span><b>{K:.1f} GPa</b></div>
            <div class="prop-row"><span>CTE α</span><b>{mat['alpha']*1e6:.1f} ×10⁻⁶/°C</b></div>
            <div class="prop-row"><span>K_IC</span><b>{mat['K_IC']/1e6:.0f} MPa√m</b></div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.subheader("📐 Plate Geometry")
        W = st.slider("Plate Width (m)", 0.5, 2.0, 1.0, 0.05, key="W")
        H = st.slider("Plate Height (m)", 0.3, 1.5, 0.5, 0.05, key="H")

        st.subheader("🕳️ Hole Configuration")
        n_holes = st.radio("Number of Holes", [1, 2, 3], horizontal=True, key="n_holes")
        holes = []
        for i in range(n_holes):
            with st.expander(f"Hole {i+1}", expanded=(i == 0)):
                rx = st.slider(f"Semi-axis Rx (m)", 0.02, 0.15, 0.05, 0.005, key=f"rx_{i}")
                ry = st.slider(f"Semi-axis Ry (m)", 0.02, 0.15, 0.05, 0.005, key=f"ry_{i}")
                cx = st.slider(f"Center X (fraction)", 0.1, 0.9, 0.5, 0.02, key=f"cx_{i}")
                cy = st.slider(f"Center Y (fraction)", 0.1, 0.9, 0.5, 0.02, key=f"cy_{i}")
                holes.append({"rx": rx, "ry": ry, "cx": cx, "cy": cy})

        st.divider()
        st.subheader("⚡ Load Configuration")
        load_type = st.selectbox("Load Type", [
            "Axial Tension", "Transverse Shear", "Biaxial Tension",
            "Compression", "Combined (Tension + Shear)",
        ], key="load_type")
        load_mag = st.slider("Load Magnitude (kN/m²)", 10.0, 1000.0, 100.0, 10.0, key="load_mag")

        st.divider()

        # RUN SIMULATION BUTTON
        run_sim = st.button("🚀  Run Simulation", use_container_width=True, type="primary")

        if run_sim:
            load_type_map = {
                "Axial Tension": "tension",
                "Transverse Shear": "shear",
                "Biaxial Tension": "biaxial",
                "Compression": "compression",
                "Combined (Tension + Shear)": "combined",
            }

            with st.spinner("Running AI simulation..."):
                # Try AI inference first, fallback to analytical
                try:
                    from src.inference import run_ai_inference
                    result = run_ai_inference(
                        geometry={"plate_width": W, "plate_height": H, "holes": holes},
                        material=mat,
                        load={
                            "type": load_type_map.get(load_type, "tension"),
                            "magnitude": load_mag * 1e3,
                            "angle": 0,
                        },
                        material_key=material_key,
                    )
                    result["source"] = "AI Surrogate"
                except Exception:
                    result = analytical_stress_field(
                        W, H, holes, material_key,
                        load_mag * 1e3,
                        load_type_map.get(load_type, "tension"),
                    )
                    # Convert numpy arrays to lists for session state
                    for k in result:
                        if hasattr(result[k], "tolist"):
                            result[k] = result[k].tolist()
                    result["source"] = "Analytical (Kirsch)"

            st.session_state["last_result"] = result
            st.session_state["last_params"] = {
                "geometry": {"plate_width": W, "plate_height": H, "holes": holes},
                "load": {"type": load_type_map.get(load_type, "tension"), "magnitude": load_mag * 1e3},
                "material_key": material_key,
            }
            st.success(f"✅ Simulation complete — {result['source']}")

    with col_preview:
        st.subheader("📐 Geometry Preview")
        fig_geo = render_geometry_preview(W, H, holes, load_type, load_mag)
        st.pyplot(fig_geo, use_container_width=True)
        plt.close(fig_geo)

        # SCF reference
        if holes:
            h0 = holes[0]
            scf_theory = scf_elliptical_hole_infinite_plate(h0["rx"], h0["ry"])
            st.info(f"📘 **Theoretical SCF** (Neuber, infinite plate): **K_t = {scf_theory:.3f}** for Rx/Ry = {h0['rx']/h0['ry']:.2f}")

# ==================== TAB 2: SIMULATION HUB ====================
with tab2:
    if "last_result" not in st.session_state:
        st.info("🎯 Configure geometry in **Design Studio** and click **Run Simulation**.")
    else:
        r = st.session_state["last_result"]

        # Metric row
        m1, m2, m3, m4, m5 = st.columns(5)
        sigma_max = r.get("sigma_max_mpa", 0)
        scf_val = r.get("scf", 0)
        sf = r.get("safety_factor", 0)
        infer_ms = r.get("inference_ms", 0) if r.get("inference_ms") else r.get("fem_time_ms", 0)

        m1.metric("Max Von Mises", f"{sigma_max:.1f} MPa")
        m2.metric("SCF (K_t)", f"{scf_val:.2f}")

        if sf > 1.5:
            m3.metric("Safety Factor", f"{sf:.2f}", delta="SAFE", delta_color="normal")
        elif sf > 1.0:
            m3.metric("Safety Factor", f"{sf:.2f}", delta="WARNING", delta_color="off")
        else:
            m3.metric("Safety Factor", f"{sf:.2f}", delta="FAILURE", delta_color="inverse")

        m4.metric("Inference Time", f"{infer_ms:.1f} ms")
        m5.metric("Source", r.get("source", "AI"))

        st.divider()

        # Visualization panels
        params = st.session_state.get("last_params", {})
        geo = params.get("geometry", {})
        vis_holes = geo.get("holes", [])
        vis_W = geo.get("plate_width", 1.0)
        vis_H = geo.get("plate_height", 0.5)

        col_ai, col_disp = st.columns(2)

        with col_ai:
            st.caption("**AI Predicted Stress (Von Mises)**")
            fig_stress = render_heatmap(
                r.get("node_coords", r.get("coords", [])),
                r.get("stress_vm", []),
                "Von Mises Stress Field",
                cmap=STRESS_CMAP,
                label="σ_vm (MPa)",
                holes=vis_holes,
                plate_w=vis_W, plate_h=vis_H,
            )
            st.pyplot(fig_stress, use_container_width=True)
            plt.close(fig_stress)

        with col_disp:
            st.caption("**Displacement Magnitude**")
            coords_arr = np.array(r.get("node_coords", r.get("coords", [[0, 0]])))
            dx = np.array(r.get("displacement_x", r.get("disp_x", [0])))
            dy = np.array(r.get("displacement_y", r.get("disp_y", [0])))
            disp_mag = np.sqrt(dx**2 + dy**2)

            fig_disp = render_heatmap(
                coords_arr, disp_mag,
                "Displacement Magnitude",
                cmap="viridis",
                label="|u| (m)",
                holes=vis_holes,
                plate_w=vis_W, plate_h=vis_H,
            )
            st.pyplot(fig_disp, use_container_width=True)
            plt.close(fig_disp)

        # FEM Comparison
        st.divider()
        st.subheader("🔄 FEM Validation & Comparison")

        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            run_fem = st.button("▶ Run FEM Validation", use_container_width=True)
        with col_info:
            st.caption("Compare AI prediction with analytical/FEM ground truth")

        if run_fem:
            with st.spinner("Running analytical solver..."):
                fem_result = analytical_stress_field(
                    vis_W, vis_H, vis_holes,
                    params.get("material_key", "structural_steel_a36"),
                    params.get("load", {}).get("magnitude", 1e5),
                    params.get("load", {}).get("type", "tension"),
                )

            col_fem, col_err = st.columns(2)
            with col_fem:
                st.caption("**FEM Ground Truth (Von Mises)**")
                fig_fem = render_heatmap(
                    fem_result["coords"], fem_result["stress_vm"],
                    "FEM Von Mises Stress",
                    cmap=STRESS_CMAP, label="σ_vm (MPa)",
                    holes=vis_holes, plate_w=vis_W, plate_h=vis_H,
                )
                st.pyplot(fig_fem, use_container_width=True)
                plt.close(fig_fem)

            with col_err:
                ai_stress = np.array(r.get("stress_vm", []))
                fem_stress = np.array(fem_result["stress_vm"])
                min_len = min(len(ai_stress), len(fem_stress))
                if min_len > 0:
                    error_pct = np.abs(ai_stress[:min_len] - fem_stress[:min_len]) / (np.abs(fem_stress[:min_len]) + 1e-8) * 100
                    st.caption("**Absolute Error Map (%)**")
                    fig_err = render_heatmap(
                        fem_result["coords"][:min_len], error_pct,
                        "Error: |AI - FEM| / FEM × 100%",
                        cmap=ERROR_CMAP, label="Error %",
                        holes=vis_holes, plate_w=vis_W, plate_h=vis_H,
                    )
                    st.pyplot(fig_err, use_container_width=True)
                    plt.close(fig_err)

                    # Error metrics
                    e1, e2, e3 = st.columns(3)
                    e1.metric("Mean Error", f"{error_pct.mean():.2f}%")
                    e2.metric("Max Error", f"{error_pct.max():.2f}%")
                    e3.metric("Median Error", f"{np.median(error_pct):.2f}%")

        # Fatigue Analysis
        st.divider()
        st.subheader("🔩 Fatigue Life Analysis")

        fat_col1, fat_col2 = st.columns([1, 2])
        with fat_col1:
            sigma_min_ratio = st.slider("Stress Ratio (R = σ_min/σ_max)", 0.0, 0.9, 0.0, 0.1, key="R_ratio")
            correction = st.selectbox("Mean Stress Correction", ["goodman", "soderberg", "gerber"], key="fat_corr")

        with fat_col2:
            sigma_max_val = r.get("sigma_max_mpa", 100)
            sigma_min_val = sigma_max_val * sigma_min_ratio

            fat_result = estimate_fatigue_life(
                sigma_max_val, sigma_min_val,
                material_key=params.get("material_key", "structural_steel_a36"),
                correction=correction,
            )

            f1, f2, f3 = st.columns(3)
            f1.metric("Cycles to Failure", f"{fat_result['cycles_to_failure']:.0f}")
            f2.metric("Life Category", fat_result["life_category"])
            f3.metric("σ_amplitude", f"{fat_result['sigma_amplitude_mpa']:.1f} MPa")

            # S-N curve
            sn = generate_sn_curve(params.get("material_key", "structural_steel_a36"))
            fig_sn, ax_sn = plt.subplots(1, 1, figsize=(8, 4), facecolor="#0f1117")
            ax_sn.set_facecolor("#1a1d27")
            ax_sn.loglog(sn["N"], sn["S"], color="#4A90D9", linewidth=2, label="S-N Curve")
            if fat_result["cycles_to_failure"] > 0 and fat_result["cycles_to_failure"] < 1e12:
                ax_sn.plot(fat_result["cycles_to_failure"], fat_result["sigma_amplitude_mpa"],
                          "o", color="#FF6B6B", markersize=10, zorder=5, label="Operating Point")
            ax_sn.axhline(y=sn["endurance_limit"], color="#3DD68C", linestyle="--", alpha=0.7, label="Endurance Limit")
            ax_sn.set_xlabel("Cycles (N)", color="#8892b0", fontsize=9)
            ax_sn.set_ylabel("Stress Amplitude (MPa)", color="#8892b0", fontsize=9)
            ax_sn.set_title(f"S-N Curve — {sn['material']}", color="#e8eaf6", fontsize=11)
            ax_sn.tick_params(colors="#8892b0", labelsize=8)
            ax_sn.legend(facecolor="#21253a", edgecolor="#2e3250", labelcolor="#e8eaf6", fontsize=8)
            ax_sn.grid(True, alpha=0.15, color="#2e3250")
            for spine in ax_sn.spines.values():
                spine.set_color("#2e3250")
            fig_sn.tight_layout()
            st.pyplot(fig_sn, use_container_width=True)
            plt.close(fig_sn)


# ==================== TAB 3: MODEL ANALYTICS ====================
with tab3:
    st.subheader("📊 Model Training Analytics")

    history_path = os.path.join("logs", "training_history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)

        # Summary metrics
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Test R² (Stress)", f"{history.get('test_r2_stress', 0):.4f}")
        a2.metric("Test R² (Disp X)", f"{history.get('test_r2_disp_x', 0):.4f}")
        a3.metric("Mean Error", f"{history.get('test_mean_rel_error_pct', 0):.2f}%")
        a4.metric("Training Time", f"{history.get('total_time_seconds', 0)/60:.1f} min")

        st.divider()

        # Loss curves
        col_loss, col_r2 = st.columns(2)

        with col_loss:
            fig_loss, ax_loss = plt.subplots(1, 1, figsize=(8, 5), facecolor="#0f1117")
            ax_loss.set_facecolor("#1a1d27")
            epochs = range(1, len(history.get("train_loss", [])) + 1)
            ax_loss.semilogy(epochs, history["train_loss"], color="#4A90D9", linewidth=1.5, label="Train Loss", alpha=0.9)
            ax_loss.semilogy(epochs, history["val_loss"], color="#FF9F1C", linewidth=1.5, label="Val Loss", alpha=0.9)
            ax_loss.set_xlabel("Epoch", color="#8892b0", fontsize=9)
            ax_loss.set_ylabel("Loss (log scale)", color="#8892b0", fontsize=9)
            ax_loss.set_title("Training & Validation Loss", color="#e8eaf6", fontsize=11, fontweight="bold")
            ax_loss.legend(facecolor="#21253a", edgecolor="#2e3250", labelcolor="#e8eaf6", fontsize=8)
            ax_loss.tick_params(colors="#8892b0", labelsize=8)
            ax_loss.grid(True, alpha=0.15, color="#2e3250")
            for spine in ax_loss.spines.values():
                spine.set_color("#2e3250")
            fig_loss.tight_layout()
            st.pyplot(fig_loss, use_container_width=True)
            plt.close(fig_loss)

        with col_r2:
            fig_r2, ax_r2 = plt.subplots(1, 1, figsize=(8, 5), facecolor="#0f1117")
            ax_r2.set_facecolor("#1a1d27")
            ax_r2.plot(epochs, history.get("val_r2_stress", []), color="#3DD68C", linewidth=2, label="Val R² (Stress)")
            ax_r2.axhline(y=0.95, color="#FF6B6B", linestyle="--", alpha=0.5, label="Target (0.95)")
            ax_r2.set_xlabel("Epoch", color="#8892b0", fontsize=9)
            ax_r2.set_ylabel("R² Score", color="#8892b0", fontsize=9)
            ax_r2.set_title("Validation R² Score", color="#e8eaf6", fontsize=11, fontweight="bold")
            ax_r2.legend(facecolor="#21253a", edgecolor="#2e3250", labelcolor="#e8eaf6", fontsize=8)
            ax_r2.tick_params(colors="#8892b0", labelsize=8)
            ax_r2.grid(True, alpha=0.15, color="#2e3250")
            ax_r2.set_ylim(0, 1.05)
            for spine in ax_r2.spines.values():
                spine.set_color("#2e3250")
            fig_r2.tight_layout()
            st.pyplot(fig_r2, use_container_width=True)
            plt.close(fig_r2)

        # Learning rate schedule
        st.subheader("📉 Learning Rate Schedule")
        fig_lr, ax_lr = plt.subplots(1, 1, figsize=(12, 3), facecolor="#0f1117")
        ax_lr.set_facecolor("#1a1d27")
        ax_lr.semilogy(epochs, history.get("lr", []), color="#A855F7", linewidth=1.5)
        ax_lr.set_xlabel("Epoch", color="#8892b0", fontsize=9)
        ax_lr.set_ylabel("Learning Rate", color="#8892b0", fontsize=9)
        ax_lr.set_title("Cosine Annealing LR Schedule", color="#e8eaf6", fontsize=11)
        ax_lr.tick_params(colors="#8892b0", labelsize=8)
        ax_lr.grid(True, alpha=0.15, color="#2e3250")
        for spine in ax_lr.spines.values():
            spine.set_color("#2e3250")
        fig_lr.tight_layout()
        st.pyplot(fig_lr, use_container_width=True)
        plt.close(fig_lr)

    else:
        st.warning("⚠️ No training history found. Run `python 2_train_ai.py` to generate training metrics.")

        # Show model architecture info
        st.subheader("🏗️ Model Architecture")
        st.code("""
Enhanced MLP with Fourier Features
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:  45 features
  ├── Coordinates (Fourier-encoded): 32 dims
  ├── Material properties: 4 dims
  ├── Geometry parameters: 4 dims
  └── Load features: 5 dims

Hidden Layers: [128 → 256 → 256 → 128]
  ├── Activation: GELU
  ├── Normalization: LayerNorm
  └── Residual Connection

Output: 3 channels
  ├── σ_vm (Von Mises stress)
  ├── u_x (displacement X)
  └── u_y (displacement Y)
        """, language="text")

        st.subheader("📋 Material Database Summary")
        mat_summary = []
        for k, v in MATERIALS.items():
            mat_summary.append({
                "Material": v["display_name"],
                "Category": v["category"],
                "E (GPa)": f"{v['E']/1e9:.1f}",
                "ν": f"{v['nu']:.3f}",
                "σ_y (MPa)": f"{v['sigma_y']/1e6:.0f}",
                "ρ (kg/m³)": v["rho"],
            })
        st.dataframe(mat_summary, use_container_width=True, hide_index=True)


# ==================== TAB 4: EXPORT & CREDITS ====================
with tab4:
    st.subheader("📄 Export Engineering Report")

    if "last_result" in st.session_state:
        st.success("✅ Simulation data available for export")

        export_format = st.selectbox("Export Format", ["PDF Report", "JSON Data", "CSV Data"])

        if st.button("📥 Generate Export", use_container_width=True):
            r = st.session_state["last_result"]
            p = st.session_state.get("last_params", {})

            if export_format == "JSON Data":
                json_str = json.dumps({
                    "simulation_result": {k: v for k, v in r.items() if not isinstance(v, (list,)) or len(v) < 100},
                    "parameters": p,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }, indent=2)
                st.download_button(
                    "⬇️ Download JSON",
                    json_str,
                    "simustruct_result.json",
                    "application/json",
                    use_container_width=True,
                )

            elif export_format == "CSV Data":
                import csv
                import io
                buf = io.StringIO()
                writer = csv.writer(buf)
                coords = r.get("node_coords", r.get("coords", []))
                stress = r.get("stress_vm", [])
                dx = r.get("displacement_x", r.get("disp_x", []))
                dy = r.get("displacement_y", r.get("disp_y", []))
                writer.writerow(["X", "Y", "Stress_VM_MPa", "Disp_X", "Disp_Y"])
                for i in range(min(len(coords), len(stress))):
                    c = coords[i] if isinstance(coords[i], (list, tuple)) else [coords[i], 0]
                    writer.writerow([
                        f"{c[0]:.6f}", f"{c[1]:.6f}",
                        f"{stress[i]:.4f}" if i < len(stress) else "",
                        f"{dx[i]:.8f}" if i < len(dx) else "",
                        f"{dy[i]:.8f}" if i < len(dy) else "",
                    ])
                st.download_button(
                    "⬇️ Download CSV",
                    buf.getvalue(),
                    "simustruct_data.csv",
                    "text/csv",
                    use_container_width=True,
                )

            elif export_format == "PDF Report":
                try:
                    from src.report import generate_report
                    pdf = generate_report(r, p, material_key=p.get("material_key", "structural_steel_a36"))
                    st.download_button(
                        "⬇️ Download PDF Report",
                        pdf,
                        "simustruct_report.pdf",
                        "application/pdf",
                        use_container_width=True,
                    )
                except ImportError:
                    st.warning("ReportLab not installed. Install: `pip install reportlab`")
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")
    else:
        st.info("Run a simulation first to enable report export.")

    # ============ DEVELOPER CREDITS ============
    st.divider()
    st.markdown("""
    <div class="credits-card">
        <h3>👨‍💻 Developer Credits</h3>
        <div style="margin-top: 16px;">
            <div class="credit-row">
                <span class="emoji">🎓</span>
                <span><b>Naman</b> — Lead Developer & AI/ML Engineer</span>
            </div>
            <div class="credit-row">
                <span class="emoji">🏛️</span>
                <span>LNMIIT Jaipur — B.Tech Project (2023-2027 Batch)</span>
            </div>
            <div class="credit-row">
                <span class="emoji">🔬</span>
                <span>SimuStruct AI V3 — Deep Learning Structural Analysis Platform</span>
            </div>
        </div>
        <hr style="border-color: #2e3250; margin: 20px 0;">
        <div style="color: #8892b0; font-size: 0.82rem;">
            <p><b>Technology Stack:</b></p>
            <p>Python • PyTorch • FastAPI • Streamlit • FEniCSx • Gmsh • Java Spring Boot</p>
            <p style="margin-top: 12px;"><b>Key Features:</b></p>
            <p>22 Engineering Materials • Fourier-Encoded MLP • Physics-Informed Training (PINN) •
            Graph Neural Network • MC Dropout Uncertainty • Fatigue Life Estimation •
            Design Optimization • Peterson's SCF Validation</p>
        </div>
        <hr style="border-color: #2e3250; margin: 20px 0;">
        <p style="color: #4A90D9; font-size: 0.8rem;">
            © 2024-2027 | Built with ❤️ at LNMIIT Jaipur
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Version info
    st.caption(f"SimuStruct AI V3.0.0 | {len(MATERIALS)} materials | PyTorch backend")
