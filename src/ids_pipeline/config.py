"""
Configuration centrale du pipeline IDS Neuro-Symbolique
(RealMLP + SHAP + Agents LLM sur logs Zeek, techniques MITRE T1046/T1595/T1587).

Toutes les valeurs peuvent être surchargées par variables d'environnement
(préfixe ``IDS_``), ce qui permet de faire tourner le pipeline sur
n'importe quelle machine — portable, serveur, CI — sans modifier le code.
"""

import os

# ==============================================================================
# 1. CHEMINS
# ==============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.environ.get("IDS_DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

MODELS_DIR = os.environ.get("IDS_MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))
MITRE_DIR = os.environ.get("IDS_MITRE_DIR", os.path.join(PROJECT_ROOT, "mitre"))
RESULTS_DIR = os.environ.get("IDS_RESULTS_DIR", os.path.join(PROJECT_ROOT, "results"))

FOCUS_DATASET_FILENAME = "dataset_focus_t1595_t1046_t1587_benin.csv"
FOCUS_DATASET_FILE = os.path.join(PROCESSED_DATA_DIR, FOCUS_DATASET_FILENAME)

REALMLP_MODEL_FILE = os.environ.get(
    "IDS_REALMLP_MODEL_FILE",
    os.path.join(MODELS_DIR, "realmlp_zeek_binaire_behavioral.joblib"),
)

MITRE_JSON_FILE = os.environ.get(
    "IDS_MITRE_JSON_FILE",
    os.path.join(MITRE_DIR, "mitre_techniques_behavioral.json"),
)

# ==============================================================================
# 2. LLM (AGENTS)
# ==============================================================================

LLM_MODEL = os.environ.get("IDS_LLM_MODEL", "gpt-oss:120b-cloud")
EMBEDDING_MODEL_NAME = os.environ.get("IDS_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ==============================================================================
# 3. TECHNIQUES CIBLÉES ET ÉCHANTILLONNAGE
# ==============================================================================

TARGET_TECHNIQUES = ["T1046", "T1595",'T1587']
# Nombre d'exemples testés par technique (Agent 1 + Agent 2 = appels LLM),
# volontairement limité à 5 pour ne pas multiplier les appels LLM. Pour
# changer le nombre d'essais : --n-per-technique sur la ligne de commande,
# ou la variable d'environnement IDS_NB_PAR_TECHNIQUE, avec la valeur voulue.
NB_PAR_TECHNIQUE = int(os.environ.get("IDS_NB_PAR_TECHNIQUE", 5))
DENSE_RETRIEVAL_TOP_K = int(os.environ.get("IDS_DENSE_TOP_K", 25))

POSSIBLE_LABEL_COLS = ["label_technique", "label_cve", "label", "target", "mitre_technique"]

XGB_KNOWN_CLASSES = ["Bénin (0)", "Attaque (1)"]

# ==============================================================================
# 4. SHAP / REGROUPEMENT HYBRIDE
# ==============================================================================

CORRELATION_THRESHOLD = 0.80
CORRELATION_SAMPLE_SIZE = 1000

# ==============================================================================
# 5. VALIDATION ADVERSARIALE (TEST D'IMPACT SHAP)
# ==============================================================================

PERMUTE_SHAP_DEFAULT = os.environ.get("IDS_PERMUTE_SHAP", "false").lower() == "true"
PERMUTATION_SEED_DEFAULT = int(os.environ.get("IDS_PERMUTATION_SEED", 42))

# ==============================================================================
# 6. LES 59 FEATURES ATTENDUES PAR REALMLP (ordre exact utilisé à l'entraînement)
# ==============================================================================


EXPECTED_FEATURES = [
    "hist_has_d", "service_gssapi,ntlm,smb", "service_ntlm,smb,gssapi,dce_rpc",
    "src_packets_sum_last_10", "dst_unique_src_last_10", "conn_state_S0",
    "resp_ip_bytes", "time_since_previous_dst_event", "service_dns",
    "conn_state_RSTRH", "resp_pkts", "hist_has_R", "hist_has_S", "hist_has_F",
    "new_destination", "service_smb,ntlm,gssapi", "service_krb_tcp",
    "hist_scan_behavior", "service_ssl", "hist_has_h", "orig_ip_bytes",
    "service_ntlm,smb,gssapi", "conn_state_OTH", "hist_has_A", "service_mysql",
    "proto_udp", "proto_tcp", "src_failed_conn_last_10",
    "service_ftp", "missed_bytes", "time_since_previous_src_event",
    "repeated_connection", "conn_state_SHR",
    "src_event_count_last_10", "conn_state_RSTO", "proto_icmp",
    "service_http",
    "conn_state_REJ", "hist_length", "service_ntp", "service_ntlm,gssapi",
    "service_gssapi", "conn_state_SH", "conn_state_S2", "duration",
    "conn_state_RSTR", "orig_bytes", "conn_state_SF", "src_duration_sum_last_10",
    "orig_pkts", "service_dce_rpc", "service_dhcp", "src_unique_dst_last_10",
    "service_gssapi,smb,ntlm", "resp_bytes", "conn_state_S1", "service_ssh",
    "src_bytes_sum_last_10", "service_gssapi,ntlm",
]

EXCLUDE_COLS_FOR_TRAINING = [
    "uid", "ts", "datetime", "community_id",
    "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "src_ip_zeek", "src_port_zeek", "dest_ip_zeek", "dest_port_zeek",
    "label", "label_technique", "label_tactic", "mitre_attack_tactics",
    "label_binary", "target", "label_cve", "vlan",
]

# ==============================================================================
# 7. DESCRIPTIONS SÉMANTIQUES DES FEATURES ZEEK (pour les agents LLM)
# ==============================================================================

FEATURE_DESCRIPTIONS = {
    "duration": "Duration of the current network connection.",
    "missed_bytes": "Number of bytes that were not successfully observed during the connection.",
    "orig_bytes": "Number of bytes sent by the source/originator.",
    "resp_bytes": "Number of bytes sent by the destination/responder.",
    "orig_ip_bytes": "Total IP-level bytes sent by the source/originator.",
    "resp_ip_bytes": "Total IP-level bytes sent by the destination/responder.",
    "orig_pkts": "Number of packets sent by the source/originator.",
    "resp_pkts": "Number of packets sent by the destination/responder.",
    "local_orig": "Indicates whether the source belongs to the local network.",
    "local_resp": "Indicates whether the destination belongs to the local network.",
    "proto_tcp": "Indicates whether the connection uses TCP.",
    "proto_udp": "Indicates whether the connection uses UDP.",
    "proto_icmp": "Indicates whether the connection uses ICMP.",
    "time_since_previous_src_event": "Time elapsed since previous event generated by the same source.",
    "time_since_previous_dst_event": "Time elapsed since previous event associated with same destination.",
    "src_event_count_last_10": "Number of events generated by same source during previous 10 events.",
    "src_packets_sum_last_10": "Total number of packets generated by same source during previous 10 events.",
    "src_bytes_sum_last_10": "Total number of bytes generated by same source during previous 10 events.",
    "src_duration_sum_last_10": "Total connection duration generated by same source during previous 10 events.",
    "src_unique_dst_last_10": "Number of different destinations contacted by same source during previous 10 events.",
    "dst_unique_src_last_10": "Number of different sources that contacted same destination during previous 10 events.",
    "repeated_connection": "Indicates whether the connection is repeated.",
    "new_destination": "Indicates whether the source is contacting a previously unseen destination.",
    "src_failed_conn_last_10": "Number of failed connection attempts generated by same source during previous 10 events.",
    "hist_scan_behavior": "Derived indicator representing a scan-like or incomplete connection pattern.",
    "hist_length": "Length of the connection history representation.",
}

# ==============================================================================
# 8. FAMILLES SÉMANTIQUES MÉTIER (pour le regroupement hybride SHAP)
# ==============================================================================

SEMANTIC_BUCKETS = {
    "Traffic_Volume": [
        "orig_bytes", "resp_bytes", "orig_ip_bytes", "resp_ip_bytes",
        "orig_pkts", "resp_pkts", "missed_bytes",
    ],
    "Flow_Dynamics": [
        "duration",
    ],
    "Source_Activity": [
        "src_event_count_last_10", "src_packets_sum_last_10", "src_bytes_sum_last_10",
        "src_duration_sum_last_10", "time_since_previous_src_event",
    ],
    "Destination_Diversity": [
        "src_unique_dst_last_10", "dst_unique_src_last_10",
    ],
    "Scanning_Behavior": [
        "src_failed_conn_last_10", "new_destination", "repeated_connection",
        "hist_scan_behavior",
    ],
    "Temporal_Behavior": [
        "time_since_previous_src_event", "time_since_previous_dst_event",
    ],
    "Protocol_and_Service": [
        "proto_tcp", "proto_udp", "proto_icmp", "service_http", "service_dns",
        "service_ssl", "service_ssh", "service_ftp", "service_dce_rpc",
        "service_dhcp", "service_mysql", "service_ntp", "service_smb,ntlm,gssapi",
        "service_krb_tcp", "service_gssapi", "service_gssapi,ntlm",
        "service_gssapi,ntlm,smb", "service_gssapi,smb,ntlm", "service_ntlm,gssapi",
        "service_ntlm,smb,gssapi", "service_ntlm,smb,gssapi,dce_rpc",
    ],
    "Connection_State": [
        "conn_state_OTH", "conn_state_REJ", "conn_state_RSTO", "conn_state_RSTR",
        "conn_state_RSTRH", "conn_state_S0", "conn_state_S1", "conn_state_S2",
        "conn_state_SF", "conn_state_SH", "conn_state_SHR",
    ],
    "Connection_History": [
        "hist_has_A", "hist_has_F", "hist_has_R", "hist_has_S", "hist_has_d",
        "hist_has_h", "hist_length", "hist_scan_behavior",
    ],
}

SERVICE_FEATURES = [
    "service_dce_rpc", "service_dhcp", "service_dns", "service_ftp", "service_gssapi",
    "service_gssapi,ntlm", "service_gssapi,ntlm,smb", "service_gssapi,smb,ntlm", "service_http",
    "service_krb_tcp", "service_mysql", "service_ntlm,gssapi", "service_ntlm,smb,gssapi",
    "service_ntlm,smb,gssapi,dce_rpc", "service_ntp", "service_smb,ntlm,gssapi", "service_ssh",
    "service_ssl",
]

CONN_STATE_FEATURES = [
    "conn_state_OTH", "conn_state_REJ", "conn_state_RSTO", "conn_state_RSTR",
    "conn_state_RSTRH", "conn_state_S0", "conn_state_S1", "conn_state_S2",
    "conn_state_SF", "conn_state_SH", "conn_state_SHR",
]

DIRECT_FEATURES = [
    "duration", "missed_bytes", "orig_bytes", "orig_ip_bytes", "orig_pkts",
    "resp_bytes", "resp_ip_bytes", "resp_pkts", "time_since_previous_src_event",
    "time_since_previous_dst_event", "src_event_count_last_10", "src_packets_sum_last_10",
    "src_bytes_sum_last_10", "src_duration_sum_last_10", "src_unique_dst_last_10",
    "dst_unique_src_last_10", "repeated_connection", "new_destination",
    "src_failed_conn_last_10",
]
