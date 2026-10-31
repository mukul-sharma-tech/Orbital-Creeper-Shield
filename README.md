# 🛰️ Anudesh Kavach: Autonomous AI Worm Defense for Satellites

**Anudesh Kavach** is a state-of-the-art prototype designed to protect satellite networks (like ISRO's NavIC or Gaganyaan missions) from self-replicating prompt injection "worms." Using a combination of Machine Learning (ML), Explainable AI (XAI), and packet analysis, it ensures the integrity of inter-satellite communication.

---

## 🚀 Overview

In modern satellite swarms, malicious payloads can hijack command protocols and propagate through the network. This project implements a robust defense layer that:
- **Detects** malicious prompt injections in real-time.
- **Quarantines** infected packets to prevent network-wide propagation.
- **Visualizes** threat intelligence through an intuitive, space-inspired dashboard.
- **Explains** detection logic using SHAP (SHapley Additive exPlanations).

---

## ✨ Key Features

- **🔴 Real-Time Detection Mode**: Simulates active satellite communication and detects injected "worms" on the fly.
- **📦 PCAP Analysis Mode**: Allows security analysts to upload and inspect standard packet capture files for past incidents.
- **🧠 ML-Powered Defense**: Uses a TF-IDF vectorizer paired with a Logistic Regression model for high-accuracy text-based threat detection.
- **🔍 Explainable AI (XAI)**: Integrated SHAP analysis to provide transparency into *why* a specific packet was flagged as malicious.
- **📊 Interactive Dashboard**: A professional, ISRO-themed Streamlit interface featuring:
  - Live threat meters and statistics.
  - Real-time detection timelines.
  - Exportable detection reports (Excel/PCAP).

---

## 🛠️ Tech Stack

- **UI/UX**: [Streamlit](https://streamlit.io/) (with custom CSS/HTML)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/), [SHAP](https://shap.readthedocs.io/)
- **Packet Processing**: [Scapy](https://scapy.net/)
- **Data Visualization**: [Plotly](https://plotly.com/), [Pandas](https://pandas.pydata.org/)
- **Serialization**: [Joblib](https://joblib.readthedocs.io/)

---

## 📋 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <project-folder>
   ```

2. **Set up a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎮 Usage

### Running the Dashboard
Launch the primary interface using Streamlit:
```bash
streamlit run app.py
```

### Running Model Training & Verification
To retrain the calibrated multi-feature model and run the edge-case verification test suite:
```bash
python model/train_enhanced_model.py
```

---

## 📁 Project Structure

- `app.py`: Main Streamlit Mission Control application with live telemetry, PCAP analysis, and report generation.
- `defense_engine.py`: Core defense engine powering the multi-feature classifier, dynamic SHAP attribution, and realistic packet simulation.
- `worm_defense_model.pkl`: Calibrated multi-feature ML pipeline (Word + Char n-grams + Security Heuristics).
- `satellite_packets.pcap`: Sample packet capture file for testing PCAP analysis mode.
- `model/data.jsonl`: Curated prompt injection and adversarial query dataset.
- `model/train_enhanced_model.py`: Standalone training and verification script for the defense pipeline.
- `model/model.ipynb`: Research notebook evaluating baseline SVM, 1D-CNN, and Quantized TinyBERT.
- `requirements.txt`: Python package dependencies.

---

## 🇮🇳 Powered by ISRO Aesthetics
This project features a custom ISRO-professional theme, incorporating typography and design elements inspired by Indian space mission control centers.

---

## 🛡️ License
This project is for educational and research purposes in the field of Satellite Cybersecurity.
