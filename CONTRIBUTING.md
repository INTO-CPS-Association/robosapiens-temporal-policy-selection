# Contributing to Temporal Policy Selection

Contributions that make the INTO-CPS navigation case study easier to understand, run, and extend are welcome.

## Scope

Please preserve the separation between the reusable temporal policy-selection logic and the task-specific Webots adapter. Changes should maintain the documented sensor observation formats, action meanings, and command-line workflows unless a breaking change is clearly documented.

## How to contribute

1. Open an issue describing the proposed improvement or problem.
2. Fork the repository and create a focused feature branch.
3. Keep code and documentation changes together; describe any new dependencies or controller options.
4. Submit a pull request explaining the changes and their effects on existing workflows.

## Development guidance

- Use descriptive Python names and docstrings for public interfaces.
- Keep anomaly-classifier data provenance and train/validation grouping explicit.
- Document any change to PPO observation routing, normalizer loading, reward logic, or safe fallback behavior.
- Identify the origin and license of any new world, PROTO, texture, dataset, or model file.
- Do not commit credentials, large generated datasets, temporary run outputs, or private checkpoints.

## Questions and bug reports

Use GitHub Issues and include the Webots version, Python environment, command used, and a concise description of the observed behavior when relevant.
