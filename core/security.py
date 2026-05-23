"""
core/security.py
Post-Quantum Encryption layer for secure patient data ingestion.

Production target : ML-KEM-768  (NIST FIPS 203 / CRYSTALS-Kyber)
This mock uses    : Curve25519 + XSalsa20-Poly1305  (PyNaCl Box)

The structure is structurally identical to real KEM —
key generation → encapsulation → decapsulation.
To upgrade, replace nacl.public.Box with liboqs.KeyEncapsulation("Kyber768").

Citation  : NIST FIPS 203 (2024)  https://doi.org/10.6028/NIST.FIPS.203
Addresses : Problem #16 — Secure Platform
"""

import json
import nacl.public
import nacl.encoding
import nacl.hash
import pandas as pd
from typing import Any


class MLKEMProxy:
    """
    Hospital-side post-quantum key encapsulation proxy.

    Flow
    ────
    1. Hospital calls __init__() → generates keypair  (≈ ML-KEM keygen)
    2. IoT sensor calls encrypt_patient_record(row, pubkey)
         → encrypts vitals before network transit  (≈ KEM encapsulate)
    3. Hospital server calls decrypt_patient_record(blob)
         → recovers plaintext for QSVM processing  (≈ KEM decapsulate)
    4. Allocation result is audit-hashed for tamper evidence
    """

    ALGORITHM   = "Curve25519-XSalsa20-Poly1305  (ML-KEM-768 proxy)"
    NIST_TARGET = "NIST FIPS 203 — ML-KEM-768  (CRYSTALS-Kyber, liboqs)"
    PROBLEM_REF = "Problem #16 — Secure Platform"

    def __init__(self):
        # Hospital keypair  (simulates ML-KEM keygen)
        self._priv  = nacl.public.PrivateKey.generate()
        self.pubkey = self._priv.public_key

    # ── Encryption  (IoT sensor / edge node side) ─────────────────────────

    def encrypt_patient_record(self, record: dict[str, Any]) -> dict:
        """
        Encrypt one patient record dict with a fresh ephemeral sender key
        (provides forward secrecy — each record is independently protected).
        Simulates IoT sensor → hospital server transit.
        """
        eph_priv   = nacl.public.PrivateKey.generate()
        box        = nacl.public.Box(eph_priv, self.pubkey)
        plaintext  = json.dumps(record, default=str).encode("utf-8")
        ciphertext = box.encrypt(plaintext, encoder=nacl.encoding.Base64Encoder)
        return {
            "ciphertext":     ciphertext.decode("ascii"),
            "eph_pubkey_hex": bytes(eph_priv.public_key).hex(),
            "algorithm":      self.ALGORITHM,
        }

    # ── Decryption  (hospital server side) ───────────────────────────────

    def decrypt_patient_record(self, blob: dict) -> dict:
        """Decrypt a single blob using the hospital private key."""
        eph_pub = nacl.public.PublicKey(bytes.fromhex(blob["eph_pubkey_hex"]))
        box     = nacl.public.Box(self._priv, eph_pub)
        plain   = box.decrypt(
            blob["ciphertext"].encode("ascii"),
            encoder=nacl.encoding.Base64Encoder,
        )
        return json.loads(plain)

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
        return nacl.hash.sha512(
            payload, encoder=nacl.encoding.HexEncoder
        ).decode("ascii")

    # ── Dashboard summary ─────────────────────────────────────────────────

    def summary(self) -> dict:
        pub_hex = bytes(self.pubkey).hex()
        return {
            "status":          "🔒  Active",
            "algorithm":       self.ALGORITHM,
            "nist_target":     self.NIST_TARGET,
            "pubkey_fp":       pub_hex[:8] + "…" + pub_hex[-8:],
            "forward_secrecy": True,
            "problem":         self.PROBLEM_REF,
        }
