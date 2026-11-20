# pipecat-aircaps

[![PyPI](https://img.shields.io/pypi/v/pipecat-aircaps)](https://pypi.org/project/pipecat-aircaps/)
[![CI](https://github.com/aircaps/pipecat-aircaps/actions/workflows/ci.yml/badge.svg)](https://github.com/aircaps/pipecat-aircaps/actions/workflows/ci.yml)

[AirCaps AI](https://aircaps.ai) speech-to-text for [Pipecat](https://github.com/pipecat-ai/pipecat).

`AirCapsLiveSTTService` streams the user's audio to **AirCaps Live 1** (`aircaps-live-1`), a low-latency English
speech-to-text model built for voice agents. It pushes interim transcripts while the user speaks, and when VAD
reports the end of speech it asks for the turn's final transcript at once. It can also let the model decide when each
turn ends (`end_of_turn=True`).

Maintained by AirCaps AI, the company that provides AirCaps Live 1. This is a Pipecat community integration.

## Installation

```bash
uv add pipecat-aircaps
```

or `pip install pipecat-aircaps`.

## Prerequisites

An AirCaps AI API key: sign in to the [AirCaps AI Console](https://console.aircaps.ai), open **API keys** and create one.
AirCaps Live 1 is available to every account, one session at a time by default
([talk to us](https://aircaps.ai/contact) for more). The [AirCaps Live 1 guide](https://aircaps.ai/docs/guides/live-1)
describes the model and its protocol.

```bash
export AIRCAPS_API_KEY=...
```

## Usage

```python
import os

from pipecat_aircaps import AirCapsLiveSTTService

stt = AirCapsLiveSTTService(api_key=os.environ["AIRCAPS_API_KEY"])

pipeline = Pipeline(
    [
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ]
)
```

Use it as you would any Pipecat STT service. It needs VAD frames to know when a turn ends, which the user aggregator
provides when it has a VAD analyzer (`LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())`), as in the example.

| Parameter | Default | |
| --- | --- | --- |
| `api_key` | required | AirCaps AI API key |
| `base_url` | `https://api.aircaps.com` | AirCaps AI API address |
| `sample_rate` | pipeline input rate | Input audio rate. Audio is resampled to 16 kHz before it is sent |
| `end_of_turn` | `False` | Let the model detect the end of each turn (see below) |
| `should_interrupt` | `True` | With `end_of_turn`: whether a new user turn interrupts the bot |
| `settings` | `model="aircaps-live-1"`, `language=Language.EN`, `end_of_turn_mode="balanced"` | `AirCapsLiveSTTService.Settings(...)` |
| `ttfs_p99_latency` | `0.25` | P99 seconds from end of speech to final transcript, for Pipecat's turn timing. Measure your own with [stt-benchmark](https://github.com/pipecat-ai/stt-benchmark) |

How it works:

- Before each connection the service asks `GET /v1/realtime/endpoint?model=aircaps-live-1` where to connect, then
  opens that WebSocket, so audio goes straight to the model.
- On `VADUserStoppedSpeakingFrame` it sends `finalize`; the final transcript of the turn comes back as one
  `TranscriptionFrame`, marked finalized, and Pipecat reports STT TTFB from it.
- If words arrive that VAD did not hear (a short or quiet word), the service asks for their final itself one second
  after the last new words, so the user turn they started still ends.
- A session the server ends (sessions last up to an hour) is reopened automatically.
- While no audio is sent (for example when the STT is muted), the service pings the session so it stays open.

## Model-detected end of turn

```python
stt = AirCapsLiveSTTService(
    api_key=os.environ["AIRCAPS_API_KEY"],
    end_of_turn=True,
    settings=AirCapsLiveSTTService.Settings(end_of_turn_mode="balanced"),
)
```

With `end_of_turn=True` the model decides when the user has finished, instead of Pipecat's VAD and turn analyzer:

- When a turn's first words arrive, the service pushes `ProposedUserStartedSpeakingFrame`.
- When the model ends the turn, it pushes the turn's final `TranscriptionFrame` (finalized), then
  `ProposedUserStoppedSpeakingFrame`.
- The service recommends `ExternalUserTurnStrategies`, so the user aggregator follows these frames. You don't need a
  VAD analyzer in the user aggregator for turn ends. If you pass your own `user_turn_strategies`, they win.

`end_of_turn_mode` trades speed for certainty: `min_latency`, `balanced` (default) or `max_accuracy`. These settings
override single values of the mode (leave them unset to keep the mode's):

| Setting | Range | `min_latency` / `balanced` / `max_accuracy` |
| --- | --- | --- |
| `end_of_turn_threshold` | 0.2 to 0.9: how sure the model must be that the turn ended | 0.3 / 0.5 / 0.6 |
| `min_turn_silence_ms` | 0 to 1000: the shortest pause that can end a turn | 40 / 80 / 400 |
| `max_turn_silence_ms` | 500 to 5000, or 0 for no limit: a pause this long always ends the turn | 2560 |
| `vad_threshold` | 0.3 to 0.9: how sure the model's voice detector must be that it hears speech | |

```python
settings = AirCapsLiveSTTService.Settings(end_of_turn_mode="balanced", min_turn_silence_ms=300)
```

All of them can be changed with `STTUpdateSettingsFrame`; the session reopens to apply them.

Not supported by AirCaps Live 1: languages other than English, key terms, speaker labels.

## Running the example

[`examples/voice-agent.py`](examples/voice-agent.py) is Pipecat's standard voice agent (OpenAI LLM, Cartesia voice)
with AirCaps AI as its speech-to-text.

```bash
git clone https://github.com/aircaps/pipecat-aircaps.git
cd pipecat-aircaps
cp .env.example .env          # add AIRCAPS_API_KEY, OPENAI_API_KEY, CARTESIA_API_KEY
uv sync --extra example
uv run examples/voice-agent.py
```

Open http://localhost:7860/client and talk to it.

## Tests

```bash
uv sync
uv run pytest
```

The tests run against a local fake of the AirCaps AI API; no key is needed.

## Compatibility

Tested with Pipecat v1.16.0, on Python 3.11, 3.12 and 3.13.

## Support

Open an [issue](https://github.com/aircaps/pipecat-aircaps/issues), or [contact AirCaps AI](https://aircaps.ai/contact).

## License

BSD 2-Clause, see [LICENSE](LICENSE).
