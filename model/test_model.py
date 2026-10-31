import sys
sys.path.insert(0, '.')
import joblib
import scapy.all as scapy
from production.defense_engine import SecurityHeuristicTransformer

model = joblib.load('worm_defense_model.pkl')
pkts = scapy.rdpcap('satellite_packets.pcap')

print(f"Total packets in PCAP: {len(pkts)}")
worm_count = 0
for i, p in enumerate(pkts):
    if p.haslayer(scapy.Raw):
        payload = p[scapy.Raw].load.decode('utf-8', errors='ignore')
        pred = model.predict([payload])[0]
        prob = model.predict_proba([payload])[0][1]
        status = "WORM DETECTED" if pred == 1 else "BENIGN"
        if pred == 1:
            worm_count += 1
        print(f"Pkt {i+1:02d} | {status:<13} (Prob: {prob:6.2%}) | {payload[:60]}")

print("-" * 60)
print(f"Total Worms Flagged in PCAP: {worm_count} / {len(pkts)}")
