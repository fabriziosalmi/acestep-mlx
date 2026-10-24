# Contributing to ACE-Step MLX

Thanks for taking the time to contribute. This project stays useful because every claim in it
is measured: a number in the README comes from a run on a real Mac, with the expectation
written down before the run. This guide covers how to report a problem, how to set up, what a
pull request needs, and how a change becomes a release.

## Ways to help

- **Report a bug**: open an [issue](https://github.com/fabriziosalmi/acestep-mlx/issues) with
  your Mac model and memory, the macOS version, the app version, what you typed, what you
  expected and what happened. The engine log, `~/Library/Logs/ACE-Step/engine.log`, usually
  explains a failure; check it before attaching it, because it contains your prompts.
- **Share listening results**: songs that came out broken, with their prompt and the seed from
  their `song.json`, help more than a general impression.
- **Send code**: for anything larger than a fix, open an issue first so we agree on the
  change before you build it.
- **Report a vulnerability**: do **not** open a public issue. Follow the
  [Security Policy](./SECURITY.md).

## Getting set up

The engine tests and `swift run` use a development install, described in the README under
[Development install and checks](./README.md#development-install-and-checks): upstream
ACE-Step 1.5 at commit `ca1e85f` with its venv and checkpoints, and this repository next to it.
Heavy runs (model loading, songs, benchmarks) belong on a Mac with 16 GB or more.

## The checks your pull request must pass

There is no CI that can load the models, so the checks run on a Mac:

```sh
cd engine && <upstream venv>/bin/python -m unittest discover -s tests
cd app && swift test               # needs Xcode (Swift Testing)
cd app && swift run AceStepE2E     # one real song through the app's model
```

What a reviewer will look for:

- **Tests that fail without the change.** A new behaviour comes with a test, and the test is
  shown to fail when the behaviour is disabled (a break-test). A test that cannot fail proves
  nothing.
- **Measurements with a prediction written first.** A change that affects speed, memory, size
  or sound adds a `bench/<step>/` folder: the prediction is committed before the run, the
  result is appended after it, raw logs go to `bench/results/`. Misses are reported, not
  rewritten.
- **Docs follow the code.** The README states only what was measured or what the code does;
  remove a claim rather than leave it unproven.
- **Small, single-topic pull requests**, with the reason in the description.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/) with a scope, for example
`feat(engine):`, `fix(app):`, `docs(bench):`, `chore(release): v0.4.0`. The subject states the
resulting behaviour, not the edit: "the engine refuses a download the disk has no room for",
not "add disk check".

## Language and style

Code, comments, commits and documentation are in English. Keep the tone plain and
professional, without emoji. Match the naming and comment density of the surrounding code.

## How a change ships

The maintainer cuts releases; a pull request does not need to touch versions. For reference:

1. Merge to `main` after the checks above pass.
2. Bump the version in `app/scripts/bundle.sh` and the README (`chore(release): vX.Y.Z`) and
   push a `vX.Y.Z` tag.
3. Build the app on the Mac mini with `app/scripts/bundle.sh`, then sign, notarize and staple
   it with `app/scripts/notarize.sh` where the Developer ID certificate is.
4. Publish a GitHub release with the zip and its SHA-256.

By contributing you agree that your contribution is licensed under the [MIT License](./LICENSE).
