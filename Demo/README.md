# CAR T-Cell Manufacturing Demo Application

This demo provides an interactive web interface for the CAR T-cell manufacturing digital twin, allowing users to visualize and compare different manufacturing strategies in real-time.

## 🎯 Overview

The demo application features:

- **Interactive Simulation**: Real-time visualization of cell culture dynamics
- **Strategy Comparison**: Compare Standard Protocol vs AI-discovered strategies
- **Patient Profiles**: Switch between different cell types based on real patient data
- **Live Metrics**: Monitor key performance indicators during simulation
- **WebSocket Streaming**: Real-time updates without page refreshes

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or navigate to the project root directory**

2. **Install main project dependencies:**
   ```bash
   pip install -r ../requirements.txt
   ```

3. **Install demo-specific dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Demo

1. **Start the web server:**
   ```bash
   python app.py
   ```

2. **Open your browser and navigate to:**
   ```
   http://127.0.0.1:5000
   ```

3. **Select a patient profile and click either:**
   - 🧪 **Run Standard Protocol**: Traditional manufacturing approach
   - 🤖 **Run AI Strategy**: AI-optimized approach using trained PPO model

## 🏗️ Architecture

### Backend (Flask + SocketIO)

- **`app.py`**: Main Flask application server
- **WebSocket Integration**: Real-time communication with frontend
- **Model Loading**: Loads pre-trained PPO model for AI strategy
- **Simulation Runner**: Executes scenarios and streams updates

### Frontend (HTML5 + Canvas)

- **`templates/index.html`**: Main web interface
- **`static/script.js`**: Frontend logic and WebSocket handling
- **`static/style.css`**: Styling and responsive design
- **Real-time Rendering**: Canvas-based visualization of cell culture

## 📊 Features

### Simulation Scenarios

1. **Standard Protocol**
   - Adds activation beads at the beginning
   - Follows traditional manufacturing steps
   - Fixed, predictable behavior

2. **AI Strategy**
   - Uses trained PPO reinforcement learning model
   - Makes dynamic decisions based on current state
   - Learns optimal bead management timing

### Patient Profiles

The demo includes patient-specific cell profiles based on real GSE data:

- **Standard Cell Profile**: Baseline parameters
- **High-Response Patient (GSE246342)**: More easily activated cells
- **Resilient Patient (GSM7867490)**: Harder to exhaust cells
- **Moderate Response Patient (GSM7867492)**: Balanced parameters

### Live Metrics

Monitor real-time KPIs:

- **Total Cells**: Current cell count in culture
- **Activated Cells**: Number of cells that have been activated
- **Potent Cells**: Cells with potency > 0.8 (highly effective)
- **Average Potency**: Mean potency across all cells

## 🔧 Configuration

### Model Path

The demo attempts to load a pre-trained model from:
```
ppo_cart_1000000.zip
```

If the model is not found, it creates a new untrained PPO model for demonstration purposes.

### Simulation Parameters

Modify patient profiles in `app.py` under `SIMULATION_PROFILES`:

```python
SIMULATION_PROFILES = {
    "Patient_GSE246342": {
        "CONVERSION_PROBABILITY": 0.15,
        "BEAD_EXHAUSTION_RATE": 0.02,
        "PROLIFERATION_PROBABILITY": 0.06,
        "NATURAL_EXHAUSTION_RATE": 0.002,
        "name": "High-Response Patient (GSE246342)"
    }
}
```

### Web Server Settings

The server runs on:
- **Host**: `0.0.0.0` (accessible from any network interface)
- **Port**: `5000`
- **Debug Mode**: Enabled by default

## 🐛 Troubleshooting

### Common Issues

1. **Model not found**
   - Ensure the trained model exists at `ppo_cart_1000000.zip`
   - Or train a new model using the main training script

2. **Port already in use**
   - Change the port in `app.py`: `socketio.run(app, ..., port=5001)`

3. **Import errors**
   - Ensure all dependencies are installed
   - Check that you're running from the Demo directory
   - Verify Python path includes parent directory

4. **WebSocket connection fails**
   - Check firewall settings
   - Ensure no proxy blocking WebSocket connections

### Performance Considerations

- The demo includes a 0.5-second delay between simulation steps for better visualization
- For faster execution, modify the sleep time in `app.py`
- Large simulations may require more memory

## 📈 Understanding the Results

### Visual Indicators

- **Red squares**: Naive (unactivated) cells
- **Blue circles**: Highly potent activated cells
- **Orange/Yellow**: Medium potency cells
- **Green circles**: Activation beads

### Performance Metrics

- **Higher potent cell count**: Better manufacturing outcome
- **Stable potency**: Indicates well-managed bead exposure
- **Total cell growth**: Shows proliferation success
