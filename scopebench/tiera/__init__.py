"""Tier-A: the private CCDA imaging tier (plan §7). Everything here is Tier-A-only (freeze_tier.py TIER_ONLY_MODULES):
it may import the shared framework but never modifies it. Patient data never leaves the enclave paths in
`scopebench.tiera.config`; manifests written to the repository carry hashed ids, row references and shifted
timestamps only (the runner reads text from the enclave copy)."""
