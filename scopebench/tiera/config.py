"""Tier-A paths (plan §1.5, §7.2). The CCDA corpus and everything derived from it with text or real dates live in the
ENCLAVE directory (inside the corpus root by default, never inside the repository). The secret salt that hashes MRNs
and drives the per-patient time shift is created once in the enclave and never committed."""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from ..config import env_path

CCDA_ROOT = env_path("SCOPEBENCH_CCDA_ROOT", Path("/path/to/authorized-ccda"))
ENCLAVE = env_path("SCOPEBENCH_TIERA_ENCLAVE", CCDA_ROOT / ".scopebench_tierA")
DUCKDB = ENCLAVE / "scopebench_ccda.duckdb"
SALT_FILE = ENCLAVE / "secret_salt.txt"
MANIFEST_TEST = ENCLAVE / "tierA"            # full manifest with text (enclave copy)
MANIFEST_DEV = ENCLAVE / "tierA_dev"
RELEASE_TEST = env_path("SCOPEBENCH_TIERA_RELEASE", ENCLAVE / "release" / "tierA")   # text-free copy
RELEASE_DEV = RELEASE_TEST.parent / "scopebench_tierA_dev"

MODALITIES = ("ctpe", "ruq_us", "fdg_pet_ct", "head_ct", "coronary_ct", "cardiac_mri", "cardiac_cath")   # seven (D6)


def secret_salt() -> str:
    """Read (or create, mode 600) the enclave salt. Raises if the enclave cannot be created."""
    if SALT_FILE.exists():
        return SALT_FILE.read_text().strip()
    ENCLAVE.mkdir(parents=True, exist_ok=True)
    os.chmod(ENCLAVE, 0o700)
    salt = secrets.token_hex(32)
    SALT_FILE.write_text(salt + "\n")
    os.chmod(SALT_FILE, 0o600)
    return salt
