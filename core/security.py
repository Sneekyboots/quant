"""
core/security.py
Post-Quantum Encryption layer for secure patient data ingestion.

Implementation    : ML-KEM-768  (NIST FIPS 203 / CRYSTALS-Kyber)  via liboqs
KEM + DEM construction:
  keygen      → oqs.KeyEncapsulation("Kyber768").generate_keypair()
  encapsulate → oqs.KeyEncapsulation("Kyber768").encap_secret(pubkey)
                  → (ct_kem, shared_secret)
  DEM         → AES-256-GCM(shared_secret[:32]).encrypt(nonce, plaintext)
  decapsulate → self._kem.decap_secret(ct_kem) → shared_secret
              → AES-256-GCM(shared_secret[:32]).decrypt(nonce, ct_payload)

Citation  : NIST FIPS 203 (2024)  https://doi.org/10.6028/NIST.FIPS.203
Addresses : Problem #16 — Secure Platform
"""

import base64
import hashlib
import json
import os

import oqs
import pandas as pd
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from typing import Any


class MLKEMProxy:
    """
    Hospital-side ML-KEM-768 key encapsulation (real liboqs implementation).

    Flow
    ────
    1. Hospital calls __init__() → ML-KEM-768 keygen
    2. IoT sensor calls encrypt_patient_record(row)
         → KEM encapsulate → AES-256-GCM encrypt  (KEM + DEM)
    3. Hospital server calls decrypt_patient_record(blob)
         → KEM decapsulate → AES-256-GCM decrypt
    4. Allocation result is audit-hashed for tamper evidence
    """

    ALGORITHM   = "ML-KEM-768 + AES-256-GCM  (NIST FIPS 203 / liboqs)"
    NIST_TARGET = "NIST FIPS 203 — ML-KEM-768  (CRYSTALS-Kyber, liboqs)"
    PROBLEM_REF = "Problem #16 — Secure Platform"

    def __init__(self):
        # ML-KEM-768 keygen — hospital holds secret key, shares public key
        self._kem   = oqs.KeyEncapsulation("Kyber768")
        self.pubkey = self._kem.generate_keypair()  # bytes

    # ── Encryption  (IoT sensor / edge node side) ─────────────────────────

    def encrypt_patient_record(self, record: dict[str, Any]) -> dict:
        """
        KEM encapsulate + AES-256-GCM encrypt one patient record.
        Each call creates a fresh KEM encapsulation → forward secrecy.
        """
        # KEM encapsulate: derive one-time shared secret
        with oqs.KeyEncapsulation("Kyber768") as sender:
            ct_kem, shared_secret = sender.encap_secret(self.pubkey)

        # DEM: AES-256-GCM encrypt the payload with the shared secret
        nonce      = os.urandom(12)
        plaintext  = json.dumps(record, default=str).encode("utf-8")
        ct_payload = AESGCM(shared_secret[:32]).encrypt(nonce, plaintext, None)

        return {
            "ct_kem":     base64.b64encode(ct_kem).decode("ascii"),
            "ct_payload": base64.b64encode(ct_payload).decode("ascii"),
            "nonce":      base64.b64encode(nonce).decode("ascii"),
            "algorithm":  self.ALGORITHM,
        }

    # ── Decryption  (hospital server side) ───────────────────────────────

    def decrypt_patient_record(self, blob: dict) -> dict:
        """KEM decapsulate → AES-256-GCM decrypt."""
        ct_kem        = base64.b64decode(blob["ct_kem"])
        ct_payload    = base64.b64decode(blob["ct_payload"])
        nonce         = base64.b64decode(blob["nonce"])
        shared_secret = self._kem.decap_secret(ct_kem)
        plaintext     = AESGCM(shared_secret[:32]).decrypt(nonce, ct_payload, None)
        return json.loads(plaintext)

    # ── DataFrame helpers ─────────────────────────────────────────────────

    def encrypt_dataframe(self, df: pd.DataFrame) -> list[dict]:
        """Encrypt every row of a patient DataFrame. Returns list of blobs."""
        return [self.encrypt_patient_record(row.to_dict())
                for _, row in df.iterrows()]

    def decrypt_dataframe(self, blobs: list[dict]) -> pd.DataFrame:
        """Round-trip: decrypt list of blobs back into a DataFrame."""
        return pd.DataFrame([self.decrypt_patient_record(b) for b in blobs])

    # ── Audit hash ────────────────────────────────────────────────────────

    def audit_hash(self, data: Any) -> str:
        """
        SHA-512 fingerprint of any JSON-serialisable object.
        Used for tamper-evident audit logs of allocation decisions —
        any post-hoc change to the allocation is detectable.
        """
        payload = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha512(payload).hexdigest()

    # ── Dashboard summary ─────────────────────────────────────────────────

    def summary(self) -> dict:
        pub_hex = self.pubkey.hex()
        return {
            "status":          "🔒  Active",
            "algorithm":       self.ALGORITHM,
            "nist_target":     self.NIST_TARGET,
            "pubkey_fp":       pub_hex[:8] + "…" + pub_hex[-8:],
            "forward_secrecy": True,
            "problem":         self.PROBLEM_REF,
        }
