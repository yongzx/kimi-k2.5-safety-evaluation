# Benchmarks

Each benchmark folder has the same public surface:

- `run_<benchmark>.sh`: one-command entrypoint for that benchmark.
- `README.md`: what paper table or figure the benchmark reproduces.
- `runner.py`: adapter used by the top-level one-button runner.
- `_impl/`: benchmark-specific implementation code and bundled static inputs.

Generated outputs are written to the configured run directory. Local smoke and paid validation outputs should go under `dump.local/`.
