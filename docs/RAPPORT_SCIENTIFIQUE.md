
# Scientific Report — Neuro-Symbolic Detection

**UWF-Zeek, RealMLP, XAI & LLM**

---

## 1. Data Engineering (Advanced Feature Engineering)

The starting point of this project was the transformation of raw
network data (Zeek logs) into a set of highly expressive features,
capable of describing the kinematics of an attack. Rather than relying
on static characteristics alone, we created complex temporal and
historical metrics.

### 1.1. Creation of temporal and geometric features

To capture dynamics (e.g., detecting fast scans or slow data
transfers), we computed the following — first taking care to sort the
flows by time and group them for each IP:

- **`pkts_per_second`**: packet sending rate;
- **`bytes_per_second`**: flow throughput;
- **`avg_bytes_per_packet`**: average size, to infer the presence of a
  payload vs. a simple handshake.

### 1.2. Creation of sliding window features

The recent history of a source or destination IP is critical. We
implemented counters over the last 10 events to identify repetitive
behaviors:

- **`src_event_count_last_10`**: number of events triggered by the
  same source;
- **`src_unique_dst_last_10`**: how many different targets the source
  has contacted (a strong indicator of sweeping);
- **`src_unique_port_last_10`**: how many different ports the source
  has scanned (a port scan indicator);
- **`src_failed_conn_last_10`**: number of connection failures,
  essential for reconnaissance detection (**T1595**, **T1046**);
- **`time_since_previous_src_event`** / **`time_since_previous_dst_event`**:
  time interval between connections, to capture request frequency.

### 1.3. Decomposition of the Zeek history and connection states

Zeek's `history` field is a compact acronym (e.g., `ShADadFf`). We
decomposed it into simple binary signals, readable by a model:

- `hist_has_S` (presence of SYN), `hist_has_h` (presence of SYN-ACK),
  `hist_has_A`, `hist_has_d`, `hist_has_F`, `hist_has_R`, etc.;
- `hist_length`: length of the history string;
- `hist_scan_behavior`: complex derived feature (boolean flag), created
  specifically to flag an interrupted or abnormal connection typical of
  a scan.

Likewise, protocols and states were one-hot encoded:

- **States**: `conn_state_S0`, `conn_state_REJ`, `conn_state_SF`, `conn_state_RSTR`, etc.;
- **Base protocols**: `proto_tcp`, `proto_udp`, `proto_icmp`;
- **Application services**: `service_http`, `service_ssl`, `service_dns`, `service_ssh`, etc.

### 1.4. Additional logical context

We kept only 4 classes for the labels: **BENIGN** traffic, **T1046**
and **T1595** and **T1587**, since the data is heavily concentrated on
these 4 categories. The other MITRE ATT&CK techniques are represented
by a much smaller number of flows, which could lead to significant
class imbalance and affect model training — a limitation also
mentioned by the dataset's creators (among the techniques detectable
from network flows alone).

We also added metrics such as `new_destination` (is the target new for
this source?) and `repeated_connection` (is the source repeatedly
hitting the same target?).

**Result of phase 1:** the final transformed dataset became a richly
detailed behavioral matrix, containing all the evidence needed for a
model (or a human) to understand the intent behind a network flow,
well beyond simple IP/Port metadata.

---

## 2. Choice and training of the learning model (RealMLP)

Rather than a classical model (Random Forest or XGBoost), we
implemented **RealMLP** via the advanced `pytabkit` framework.

**Why RealMLP?** This multilayer neural network, specialized for
tabular data, proved extremely effective at capturing the complex
non-linear relationships created by our dozens of temporal and
historical features.




## 3. Semantic explainability (SHAP) and multi-agent LLMs

Since the RealMLP model is a black box, **SHAP** is used to extract
feature importance for each network event. To avoid overwhelming Agent
1 with raw statistics, these SHAP values are aggregated into **business
semantic families**: *Velocity and Frequency*, *Volume and Duration*,
*Exploration and Reconnaissance*, *Protocols and Services*, *Connection
State and Fails* (see `src/ids_pipeline/config.py::SEMANTIC_BUCKETS`
for the exact implementation — the grouping combines these families
with a DFS over Pearson correlation, |r| ≥ 0.80).

### Inference architecture developed

1. **Agent 1 (behavioral synthesizer)** — receives the raw log and the
   SHAP semantic explainability, and generates a summary
   (`EVENT SUMMARY`) and a tactical description (`BEHAVIOR
   DESCRIPTION`).
2. **Dense Retrieval (MITRE ATT&CK RAG)** — encodes the textual
   description via `SentenceTransformer` (`all-MiniLM-L6-v2`) and
   retrieves the candidate MITRE techniques from the JSON knowledge
   base.
3. **Agent 2 (final classifier)** — analyzes the candidate techniques
   and ranks the 5 most likely ones in JSON format.

