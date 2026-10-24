# Security Policy

ACE-Step MLX is a local macOS app: songs are made on your Mac, there is no backend, no
account and no telemetry. The app talks to the network once, on the first start, to
download the model weights from Hugging Face; after that it works offline.

## Reporting a vulnerability

**Please do not open a public issue for security reports.**

- Preferred: open a private report via **GitHub Security Advisories**:
  [Report a vulnerability](https://github.com/fabriziosalmi/acestep-mlx/security/advisories/new).
- Alternatively, email **fabrizio.salmi@gmail.com** with the details.

Please include:

- the app version (the release tag, or ACE-Step MLX > About), your macOS version and Mac model;
- a description of the issue and its impact;
- steps to reproduce, a proof of concept, or the relevant part of
  `~/Library/Logs/ACE-Step/engine.log` (check it first: it contains your prompts).

This is a solo-maintained project, so responses are best-effort: expect an acknowledgement
within a few days. Please allow a reasonable window for a fix before any public disclosure.
Coordinated disclosure is appreciated, and reporters are credited in the release notes unless
they prefer to stay anonymous.

## Supported versions

Security fixes land on the latest release. Please reproduce on the most recent release before
reporting.

| Version | Supported |
| ------- | --------- |
| latest `0.x` release | Yes |
| older releases | No |

## Scope and hardening

What the app already does; a report that defeats one of these is especially valuable:

- **Distribution**: the release zip is signed with a Developer ID certificate, every binary
  inside runs with the hardened runtime and no entitlements, and the app is notarized and
  stapled by Apple. The release notes give the zip's SHA-256.
- **Model download**: weights come from Hugging Face at pinned revisions; the sha256 of every
  weight file is checked against the repository's metadata before it is used, and a
  download that does not match is refused. The disk space is checked before anything is
  fetched.
- **Dependencies**: the Python packages inside the app are installed from upstream's lock file
  with hashes required; packages no engine path imports are removed.
- **Engine process**: the window talks to the engine over the process's own pipes; the engine
  opens no listening socket. The bytecode cache lives outside the app, so the signed bundle
  is never written to.
- **Requests**: options are validated by the engine (types, ranges, at most 4000 characters
  of lyrics); "another like this" only resolves a song folder inside the songs directory and
  refuses names or symlinks that lead out of it.

### Out of scope

- Issues that require a previously compromised machine or physical access.
- The content of generated songs.
- Vulnerabilities in ACE-Step itself or in third-party packages without a demonstrated impact on
  this app: please report those upstream, though a heads-up is welcome.

Thank you for helping keep ACE-Step MLX and its users safe.
