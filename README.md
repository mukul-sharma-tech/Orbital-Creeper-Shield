# 🛰️ Orbital Creeper Shield: Autonomous AI Worm Defense for Satellites

**Orbital Creeper Shield** is a state-of-the-art prototype designed to protect satellite networks (like ISRO's NavIC or Gaganyaan missions) from self-replicating prompt injection "worms." Using a combination of Machine Learning (ML), Explainable AI (XAI), and packet analysis, it ensures the integrity of inter-satellite communication.

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
streamlit run Project/interface.py
```

### Running the Prototype Script
To see the core logic (packet generation and model training) in a CLI format:
```bash
python app.py
```

---

## 📁 Project Structure

- `Project/interface.py`: The main Streamlit application with the full UI.
- `app.py`: The core prototype engine for packet generation, model training, and SHAP results.
- `requirements.txt`: List of Python dependencies.
- `Project/satellite_packets.pcap`: Sample packet capture file for testing.
- `Project/worm_defense_model.pkl`: The serialized ML model.
- `realtime_section.py` / `replace_section.py`: Utility scripts for UI modularity and maintenance.

---

## 🇮🇳 Powered by ISRO Aesthetics
This project features a custom ISRO-professional theme, incorporating typography and design elements inspired by Indian space mission control centers.

---

## 🛡️ License
This project is for educational and research purposes in the field of Satellite Cybersecurity.
