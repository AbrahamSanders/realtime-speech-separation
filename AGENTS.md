# Repository instructions

## Start here

- Read [README.md](README.md) for motivation, setup & usage instructions,
  and other front-line project information.
- Follow [docs/architecture.md](docs/architecture.md). Code goes in
  `src/realtime_speech_separation/`; pytest tests go in `tests/`.
- Keep code and documentation simple, readable, and easy to change.
  Use plain language and add structure only when it helps.
- Keep evaluation metric definitions and motivations in
  [docs/evaluation-protocol.md](docs/evaluation-protocol.md).

## Development environment

- Use the Miniconda environment `audio` for all local Python work.
  Create it with
  `conda create --name audio python=3.12`.
  The README has package installation steps; local development uses Python 3.12.
- Use conda for environments and the environment's `python -m pip` for packages.
- For noninteractive commands, use
  `conda run --no-capture-output -n audio python ...`.

### Dependency Directories:
- `MagiCodec`: contains the MagiCodec repo clone, since it has no pip-installable package.
  Do not modify anything in this directory directly.
- `data`: contains source audio and codec-encoded versions. Note - this is likely a symlink to a shared storage location used by many projects.
  Do not modify anything in this directory without express permission. You are only concerned with the following subdirectories:
  * `data/audio/raw`: contains the raw source audio files.
  * `data/audio/codes`: contains the codec-encoded versions of the source audio files.
  * `data/vads`: contains voice activity detection (VAD) information for the source audio files, extracted with Silero VAD.

## Code and checks

- Prefer small functions and clear interfaces. Use type hints for public functions
  and document audio shapes, units, and error behavior where relevant.
- Add dependencies when a feature needs them. Take adavantage of established
  libraries and packages when available, rather than re-inventing routines from scratch.
- Use comments sparingly - comment only when the code contains non-obvious assumptions
  or readibility would benefit from the motivation of a specific decision. Always comment:
  * The main steps of any process or algorithm
  * Docstrings on functions, classes, and modules - purpose, usage, and intended consumers.
- Write focused pytest tests. Keep most unit tests offline and CPU-only, using small
  synthetic inputs and fixed seeds where needed. GPU-bound operations such as
  separation inference may also be tested automatically, but these tests should be
  run only on-demand, not with common CI checks or after changing unrelated areas of the code.
- In the project environment, run `python -m ruff check .`,
  `python -m ruff format --check .`, and `python -m pytest`.

## GPU etiquette

- Assume you are working on a development server with shared GPU resources.
  before running anything on a GPU (inference, smoke tests, GPU-bound unit tests)
  make sure to run `nvidia-smi` first. Select the GPU with no (or least) usage and
  restrict your run to it using the `CUDA_VISIBLE_DEVICES` environment variable.
- It is possible at any time that there will be insufficient GPU capacity to run a
  GPU-bound process, in which case you will see CUDA OOM errors. Unless otherwise
  specified, this means that you need to wait and try again later.

## Review and documentation

- Check behavior, edge cases, and dependency changes. Describe what changed and
  which checks ran; report any checks that could not run.
- Update the relevant document and link to it rather than copying instructions.
- Keep README files environment-manager neutral: use plain `python` commands
  without the `conda run --no-capture-output -n audio` prefix or
  conda-specific setup instructions. Keep agent environment instructions here
  or in other agent-facing documentation.
- Do not commit to git unless explicitly requested.
