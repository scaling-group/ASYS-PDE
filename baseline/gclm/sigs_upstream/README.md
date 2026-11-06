# SIGS decoder

The inference source in `upstream/src/sigs` is from
[oroikono/SIGS](https://github.com/oroikono/SIGS/tree/b7e5f6d44f0b1dd39a64aaf4e4a3dd30d5497140),
revision `b7e5f6d44f0b1dd39a64aaf4e4a3dd30d5497140`.
The upstream [MIT license](upstream/LICENSE) applies to these files.

From the repository root:

```bash
uv sync --extra baseline
baseline/gclm/sigs_upstream/fetch_assets.sh
```

The script retrieves the original cluster library (4.6 MB) and pretrained
checkpoint (73.5 MB) from that revision, checks their SHA-256 hashes, and saves
them in the ignored `assets/` directory. Verified existing files are reused;
downloads replace their destination only after successful verification.
