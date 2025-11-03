# CAR T-cell Digital Twin - Calibration Guide

This guide explains how to calibrate the CAR T-cell digital twin environment to match your experimental data.

## 🎯 **Target Metrics**

### **Yield Targets**
- **G-Rex**: 50-100x expansion over 10-12 days
- **Stirred Tank**: 100-200x expansion over 10-12 days
- **Starting density**: 0.5-2.0 × 10⁶ cells/mL

### **Phenotype Targets**
- **G-Rex**: <15% exhaustion fraction
- **Stirred Tank**: <20% exhaustion fraction
- **Memory cells**: Prefer Tcm/Tef over exhausted cells

### **Process Sanity**
- **Glucose**: >5 mM (avoid starvation)
- **Lactate**: <20 mM (avoid toxicity)
- **Resource usage**: Bounded IL-2 and media consumption

## 🔧 **Calibration Process**

### **Step 1: Run Calibration Guide**
```bash
python calibration_guide.py
```

This will:
- Run baseline simulations with current parameters
- Show expansion curves, nutrient dynamics, and exhaustion
- Suggest parameter adjustments based on targets

### **Step 2: Input Your Data**
Edit `calibration_template.py` with your experimental data:

```python
target_data = {
    'days': [0, 4, 7, 10, 12],  # Your time points
    'cell_counts': [2.0, 15.0, 45.0, 80.0, 100.0],  # Your cell counts
    'glucose': [25.0, 20.0, 15.0, 10.0, 8.0],  # Your glucose data
    'lactate': [2.0, 5.0, 10.0, 15.0, 18.0],  # Your lactate data
    'exhaustion': [0.05, 0.08, 0.12, 0.15, 0.18],  # Your exhaustion data
    'il2': [0, 50, 100, 80, 60]  # Your IL-2 schedule
}
```

### **Step 3: Adjust Parameters**
Modify `PLATFORMS` in `car_t_env.py` based on suggestions:

```python
PLATFORMS = {
    "grex": dict(
        r_base=0.25,           # Adjust growth rate
        K=400.0,               # Adjust carrying capacity
        glc_consume=0.20,      # Adjust glucose consumption
        lac_prod=0.18,         # Adjust lactate production
        exhaustion_base=0.001, # Adjust baseline exhaustion
        # ... other parameters
    )
}
```

### **Step 4: Verify Calibration**
Re-run calibration to verify improvements match your data.

## 📊 **Key Parameters to Calibrate**

### **Growth Parameters**
- `r_base`: Baseline growth rate (0.2-0.4)
- `K`: Carrying capacity (300-800 × 10⁶ cells)
- `o2_bonus`: Oxygen bonus (0.08-0.15)

### **Nutrient Parameters**
- `glc_consume`: Glucose consumption rate (0.15-0.35)
- `lac_prod`: Lactate production rate (0.15-0.30)

### **Exhaustion Parameters**
- `exhaustion_base`: Baseline exhaustion (0.001-0.005)
- `exhaustion_activation`: Stimulation-induced exhaustion (0.010-0.020)
- `exhaustion_il2`: IL-2-induced exhaustion (0.005-0.010)

### **Platform Differences**
- **G-Rex**: Lower K, higher O₂ bonus, no shear penalty
- **Stirred Tank**: Higher K, agitation-dependent O₂, shear penalty

## 🧬 **T-cell Subpopulations**

For more realistic modeling, use `tcell_subpopulations.py`:

```python
from tcell_subpopulations import TCellSubpopulationEnv

env = TCellSubpopulationEnv(platform="grex")
```

This adds:
- **Tn**: Naive T cells
- **Tcm**: Central memory T cells  
- **Tef**: Effector T cells
- **Tex**: Exhausted T cells

## 📈 **Typical Experimental Schedules**

### **G-Rex Platform**
- **Seed density**: 0.5-1.0 × 10⁶ cells/mL
- **IL-2 schedule**: 50-100 IU/mL days 0-3, 25-50 IU/mL days 4-7
- **Expected expansion**: 50-100x over 10-12 days
- **Target exhaustion**: <15%

### **Stirred Tank Platform**
- **Seed density**: 0.5-1.0 × 10⁶ cells/mL
- **IL-2 schedule**: 100-200 IU/mL days 0-3, 50-100 IU/mL days 4-7
- **Expected expansion**: 100-200x over 10-12 days
- **Target exhaustion**: <20%

## 🔍 **Calibration Checklist**

- [ ] Run baseline calibration
- [ ] Input your experimental data
- [ ] Adjust growth parameters (r_base, K)
- [ ] Adjust nutrient parameters (glc_consume, lac_prod)
- [ ] Adjust exhaustion parameters
- [ ] Verify expansion targets
- [ ] Check process sanity (glucose >5mM, lactate <20mM)
- [ ] Test with PPO training
- [ ] Evaluate against targets

## 🚀 **Training with Calibrated Parameters**

After calibration:

```bash
# Train PPO agent
python train_ppo.py

# Evaluate trained model
python evaluate_model.py ppo_cart_grex grex 10
```

The evaluation will show:
- **Yield performance**: Expansion ratio vs targets
- **Phenotype quality**: Exhaustion fraction vs targets  
- **Process stability**: Glucose/lactate violations
- **Resource efficiency**: IL-2 and media usage

## 📝 **Example Calibration Results**

```
TARGET EVALUATION - GREX PLATFORM
============================================================
YIELD TARGETS:
  Target expansion: 50x
  Achieved expansion: 45.2x (range: 38.1-52.3)
  Target met: True (80% of target)

PHENOTYPE TARGETS:
  Target exhaustion: <15%
  Achieved exhaustion: 12.3% ± 2.1%
  Target met: True

PROCESS SANITY:
  Glucose <5 mM: 2.1 steps/episode
  Lactate >20 mM: 1.8 steps/episode
  Process stable: True

OVERALL PERFORMANCE SCORE: 0.85/1.0
  Yield: 0.90
  Phenotype: 0.82
  Process: 0.83
```

## 🎯 **Next Steps**

1. **Calibrate to your data**: Use the calibration guide
2. **Train PPO agents**: Optimize control strategies
3. **Evaluate performance**: Check against biological targets
4. **Extend modeling**: Add T-cell subpopulations
5. **Validate clinically**: Test with real manufacturing data
