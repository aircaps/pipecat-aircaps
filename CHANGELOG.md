# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.1] - 2026-11-20

### Changed

- The company is named AirCaps AI in the README, docstrings, log and error messages, and the package metadata.
  No change in behavior.

## [0.1.0] - 2026-11-19

### Added

- `AirCapsLiveSTTService`: streaming speech-to-text with AirCaps Live 1 (`aircaps-live-1`). Interim transcripts while
  the user speaks; a final transcript per turn, asked for when VAD reports the end of speech; endpoint lookup before
  each connection; automatic reconnection; keepalive pings while no audio is sent.
- `end_of_turn=True`: the model detects the end of each turn. The service pushes
  `ProposedUserStartedSpeakingFrame` / `ProposedUserStoppedSpeakingFrame` around each turn's final transcript and
  recommends `ExternalUserTurnStrategies`. A new turn's first transcript waits (up to 0.5 s) until Pipecat has started
  the turn. `end_of_turn_mode` setting (`min_latency`, `balanced`, `max_accuracy`), and `end_of_turn_threshold`,
  `min_turn_silence_ms`, `max_turn_silence_ms`, `vad_threshold` to override single values of the mode.
- Words VAD did not hear are finalized one second after the last new words, so the user turn they started ends.
- Text that is only punctuation is never pushed; punctuation that closes the previous turn is not repeated at the
  start of the next one.
- Foundational example `examples/voice-agent.py`.
- Tests against a local fake of the AirCaps API.

Tested with Pipecat v1.16.0.

[Unreleased]: https://github.com/aircaps/pipecat-aircaps/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/aircaps/pipecat-aircaps/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/aircaps/pipecat-aircaps/releases/tag/v0.1.0
