#
# Copyright (c) 2026, AirCaps
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests against a local fake of the AirCaps endpoint lookup and live session."""

import asyncio
import json
import time

from aiohttp import WSMsgType, web
from pipecat.frames.frames import (
    ErrorFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.errors import ErrorCategory

from pipecat_aircaps import AirCapsLiveSTTService

KEY = "test-key"


class FakeAirCaps:
    """The endpoint lookup and a session that says one line per turn.

    Like the real service, every transcript event carries the session's whole
    text, and a ``finalize`` makes everything so far final. With end of turn on
    in the start message, the fake ends a turn itself once it has 0.2 s of
    audio, with a final carrying ``end_of_turn``.
    """

    def __init__(self, turns, *, lookup_status=200, end_session_after_turn=None, cut_after=None):
        self.turns = list(turns)
        self.lookup_status = lookup_status
        self.end_session_after_turn = end_session_after_turn
        self.cut_after = cut_after  # close the session after this many audio bytes
        self.sessions = []  # one dict per session: start message, controls, audio bytes
        self.server = None

    async def __aenter__(self):
        app = web.Application()
        app.router.add_get("/v1/realtime/endpoint", self._lookup)
        app.router.add_get("/v1/realtime", self._session)
        self.server = web.AppRunner(app)
        await self.server.setup()
        site = web.TCPSite(self.server, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        return self

    async def __aexit__(self, *args):
        await self.server.cleanup()

    async def _lookup(self, request):
        assert request.headers["Authorization"] == f"Bearer {KEY}"
        assert request.query["model"] == "aircaps-live-1"
        if self.lookup_status != 200:
            error = {"code": "model_access_denied", "message": "No access to this model."}
            return web.json_response({"error": error}, status=self.lookup_status)
        return web.json_response(
            {"websocket_url": f"ws://127.0.0.1:{self.port}/v1/realtime", "model": "aircaps-live-1"}
        )

    async def _session(self, request):
        assert request.headers["Authorization"] == f"Bearer {KEY}"
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        session = {"start": None, "controls": [], "audio": 0}
        self.sessions.append(session)
        text = ""
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                assert session["start"], "audio before start"
                first = session["audio"] == 0
                session["audio"] += len(msg.data)
                if first and self.turns:
                    first = self.turns[0]
                    words = (first[0] if isinstance(first, tuple) else first).split()
                    interim = " ".join(words[: max(1, len(words) // 2)])
                    await ws.send_json(_transcript(_join(text, interim), False))
                if self.cut_after and session["audio"] >= self.cut_after:
                    self.cut_after = None
                    await ws.close()
                elif session["start"].get("end_of_turn") and session["audio"] >= 6400:
                    if self.turns:
                        turn = self.turns.pop(0)
                        # a turn given as (text, "max_utterance"): a final that does not end the turn
                        turn, reason = turn if isinstance(turn, tuple) else (turn, "model")
                        text = _join(text, turn)
                        session["audio"] = 0
                        end = {
                            "end_of_turn": reason == "model",
                            "reason": reason,
                            "end_of_turn_confidence": 0.9,
                        }
                        await ws.send_json({**_transcript(text, True), **end})
            elif msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data["type"] == "start":
                    session["start"] = data
                    await ws.send_json({"type": "provider_status", "status": "connecting"})
                    await ws.send_json({"type": "ready"})
                    continue
                session["controls"].append(data["type"])
                if data["type"] == "finalize" and self.turns:
                    text = _join(text, self.turns.pop(0))
                    session["audio"] = 0
                    await ws.send_json(_transcript(text, True))
                    if self.end_session_after_turn == len(session["controls"]):
                        self.end_session_after_turn = None
                        await ws.send_json({"type": "session_limit", "message": "limit"})
                        await ws.close()
                elif data["type"] == "finish":
                    await ws.send_json({"type": "finished"})
                    await ws.close()
        return ws


def _join(text, more):
    return f"{text} {more}" if text else more


def _transcript(text, is_final):
    return {"type": "transcript", "text": text, "final_chars": len(text), "is_final": is_final}


def _speech(sample_rate=16000, seconds=0.2):
    frame = 320 * sample_rate // 16000
    return [
        InputAudioRawFrame(audio=b"\x01\x00" * frame, sample_rate=sample_rate, num_channels=1)
        for _ in range(int(seconds / 0.02))
    ]


def _turn(sample_rate=16000):
    return [
        VADUserStartedSpeakingFrame(start_secs=0.2, timestamp=time.time()),
        *_speech(sample_rate),
        SleepFrame(0.2),
        VADUserStoppedSpeakingFrame(stop_secs=0.2, timestamp=time.time()),
        SleepFrame(0.3),
    ]


def _texts(frames, kind):
    return [f.text for f in frames if type(f) is kind]


async def test_turns_become_one_final_each():
    async with FakeAirCaps(["Hi there.", "Book a table for two."]) as fake:
        stt = AirCapsLiveSTTService(api_key=KEY, base_url=fake.base_url)
        down, _ = await run_test(stt, frames_to_send=[*_turn(), *_turn()])

    assert _texts(down, TranscriptionFrame) == ["Hi there.", "Book a table for two."]
    assert _texts(down, InterimTranscriptionFrame) == ["Hi", "Book a"]
    finals = [f for f in down if type(f) is TranscriptionFrame]
    assert all(f.finalized for f in finals)

    [session] = fake.sessions
    assert session["start"] == {
        "type": "start",
        "model": "aircaps-live-1",
        "encoding": "pcm_s16le",
        "sample_rate": 16000,
        "channels": 1,
    }
    assert session["controls"][:2] == ["finalize", "finalize"]


async def test_audio_is_sent_at_16_khz():
    async with FakeAirCaps([]) as fake:
        stt = AirCapsLiveSTTService(api_key=KEY, base_url=fake.base_url)
        await run_test(stt, frames_to_send=[*_speech(48000, seconds=1.0), SleepFrame(0.3)])

    # one second of 16-bit mono audio at 16 kHz, within the resampler's latency
    assert 30000 <= fake.sessions[0]["audio"] <= 32000


async def test_a_new_session_after_the_server_ends_one():
    async with FakeAirCaps(["First turn.", "Second turn."], end_session_after_turn=1) as fake:
        stt = AirCapsLiveSTTService(api_key=KEY, base_url=fake.base_url)
        down, _ = await run_test(stt, frames_to_send=[*_turn(), SleepFrame(0.5), *_turn()])

    assert len(fake.sessions) == 2
    assert _texts(down, TranscriptionFrame) == ["First turn.", "Second turn."]


async def test_lookup_refused_is_reported():
    async with FakeAirCaps([], lookup_status=403) as fake:
        stt = AirCapsLiveSTTService(api_key=KEY, base_url=fake.base_url)
        _, up = await run_test(stt, frames_to_send=[SleepFrame(0.2)])

    errors = [f for f in up if isinstance(f, ErrorFrame)]
    # one error, and no reconnect attempts: a refused model is permanent
    assert len(errors) == 1 and "model_access_denied" in errors[0].error
    assert errors[0].category == ErrorCategory.AUTHORIZATION
    assert not fake.sessions


async def test_keepalive_is_a_ping_not_audio():
    sent = []

    class Socket:
        async def send(self, message):
            sent.append(message)

    stt = AirCapsLiveSTTService(api_key=KEY)
    stt._websocket = Socket()
    await stt._send_keepalive(b"\x00" * 640)
    assert sent == ['{"type": "ping"}']


TURN_FRAMES = (
    ProposedUserStartedSpeakingFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    ProposedUserStoppedSpeakingFrame,
)


def _turn_events(frames):
    return [(type(f).__name__, getattr(f, "text", None)) for f in frames if type(f) in TURN_FRAMES]


class TurnStarter(FrameProcessor):
    """Stands in for Pipecat's user aggregator after the STT service: it starts each
    proposed user turn, sending UserStartedSpeakingFrame back upstream ``delay``
    seconds later, and notes "started" on the timeline."""

    def __init__(self, timeline: list, delay: float = 0.0):
        super().__init__()
        self._timeline = timeline
        self._delay = delay

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(
            frame, ProposedUserStartedSpeakingFrame
        ):
            self.create_task(self._start_turn())

    async def _start_turn(self):
        await asyncio.sleep(self._delay)
        self._timeline.append("started")
        await self.push_frame(UserStartedSpeakingFrame(), FrameDirection.UPSTREAM)


def _with_turns(stt, delay: float = 0.0):
    """The service followed by a TurnStarter, and the timeline: the turn frames the
    service pushes downstream, in the order it pushes them, and each turn start.
    (Further on, a system frame such as a proposed turn start can overtake data
    frames still queued, so the frames that reach the end are not always in this
    order.)"""
    timeline = []
    push = stt.push_frame

    async def push_and_note(frame, direction=FrameDirection.DOWNSTREAM):
        if direction == FrameDirection.DOWNSTREAM and type(frame) in TURN_FRAMES:
            timeline.append(frame)
        await push(frame, direction)

    stt.push_frame = push_and_note
    return Pipeline([stt, TurnStarter(timeline, delay)]), timeline


async def test_end_of_turn_lets_the_model_end_turns():
    async with FakeAirCaps(["Hi there.", "Book a table for two."]) as fake:
        stt = AirCapsLiveSTTService(api_key=KEY, base_url=fake.base_url, end_of_turn=True)
        # no VAD frames: the model decides where turns end
        speech = [*_speech(seconds=0.3), SleepFrame(0.3)]
        pipeline, timeline = _with_turns(stt)
        down, _ = await run_test(pipeline, frames_to_send=[*speech, *speech], ignore_start=False)

    assert _turn_events(timeline) == [
        ("ProposedUserStartedSpeakingFrame", None),
        ("InterimTranscriptionFrame", "Hi"),
        ("TranscriptionFrame", "Hi there."),
        ("ProposedUserStoppedSpeakingFrame", None),
        ("ProposedUserStartedSpeakingFrame", None),
        ("InterimTranscriptionFrame", "Book a"),
        ("TranscriptionFrame", "Book a table for two."),
        ("ProposedUserStoppedSpeakingFrame", None),
    ]
    assert all(f.finalized for f in down if type(f) is TranscriptionFrame)

    [session] = fake.sessions
    assert session["start"]["end_of_turn"] is True
    assert session["start"]["end_of_turn_mode"] == "balanced"
    assert "finalize" not in session["controls"]

    [metadata] = [f for f in down if isinstance(f, STTMetadataFrame)]
    assert isinstance(metadata.user_turn_strategies, ExternalUserTurnStrategies)
    assert metadata.ttfs_p99_latency == 0


async def test_end_of_turn_mode_setting():
    async with FakeAirCaps([]) as fake:
        stt = AirCapsLiveSTTService(
            api_key=KEY,
            base_url=fake.base_url,
            end_of_turn=True,
            settings=AirCapsLiveSTTService.Settings(end_of_turn_mode="min_latency"),
        )
        await run_test(stt, frames_to_send=[SleepFrame(0.2)])

    assert fake.sessions[0]["start"]["end_of_turn_mode"] == "min_latency"
    for name in ("end_of_turn_threshold", "min_turn_silence_ms", "max_turn_silence_ms"):
        assert name not in fake.sessions[0]["start"]


async def test_end_of_turn_values_override_the_mode():
    async with FakeAirCaps([]) as fake:
        stt = AirCapsLiveSTTService(
            api_key=KEY,
            base_url=fake.base_url,
            end_of_turn=True,
            settings=AirCapsLiveSTTService.Settings(
                end_of_turn_threshold=0.6, min_turn_silence_ms=200, vad_threshold=0.5
            ),
        )
        await run_test(stt, frames_to_send=[SleepFrame(0.2)])

    start = fake.sessions[0]["start"]
    assert start["end_of_turn_mode"] == "balanced"
    assert (start["end_of_turn_threshold"], start["min_turn_silence_ms"]) == (0.6, 200)
    assert start["vad_threshold"] == 0.5 and "max_turn_silence_ms" not in start


async def test_end_of_turn_ends_a_turn_cut_short():
    async with FakeAirCaps(["Hi there."], cut_after=3200) as fake:
        stt = AirCapsLiveSTTService(api_key=KEY, base_url=fake.base_url, end_of_turn=True)
        pipeline, timeline = _with_turns(stt)
        down, _ = await run_test(pipeline, frames_to_send=[*_speech(seconds=0.2), SleepFrame(0.5)])

    # the session closed mid-turn: its text so far, not finalized, then the turn's end
    assert _turn_events(timeline)[-2:] == [
        ("TranscriptionFrame", "Hi"),
        ("ProposedUserStoppedSpeakingFrame", None),
    ]
    assert not [f for f in down if type(f) is TranscriptionFrame][-1].finalized


async def test_without_end_of_turn_pipecat_decides():
    stt = AirCapsLiveSTTService(api_key=KEY)
    assert stt.supports_ttfs
    assert stt.service_metadata_frame().user_turn_strategies is None


async def test_end_of_turn_punctuation_alone_is_not_a_turn():
    stt = AirCapsLiveSTTService(api_key=KEY, end_of_turn=True)
    pushed = []

    async def push(frame, direction=None):
        pushed.append(type(frame).__name__)

    stt.push_frame = stt.broadcast_frame = push
    stt._text, stt._final_length = "Hi there", len("Hi there")
    await stt._on_transcript({"type": "transcript", "text": "Hi there?", "is_final": False})
    assert "ProposedUserStartedSpeakingFrame" not in pushed and not stt._in_turn


async def test_punctuation_alone_is_not_a_transcript():
    """After a final, the model can add "?" to it; that must not start a user turn."""
    stt = AirCapsLiveSTTService(api_key=KEY)
    pushed = []

    async def push(frame, direction=None):
        pushed.append(type(frame).__name__)

    stt.push_frame = stt.broadcast_frame = push
    stt._text, stt._final_length = "Can you hear me", len("Can you hear me")
    await stt._on_transcript({"type": "transcript", "text": "Can you hear me?", "is_final": False})
    await stt._on_transcript({"type": "transcript", "text": "Can you hear me?", "is_final": True})
    assert pushed == []


async def test_punctuation_closing_the_previous_turn_is_not_in_the_next():
    stt = AirCapsLiveSTTService(api_key=KEY)
    stt._restart_missed_speech_timer = lambda: None
    pushed = []

    async def push(frame, direction=None):
        pushed.append(frame.text)

    stt.push_frame = push
    stt._text, stt._final_length = "tomorrow evening", len("tomorrow evening")
    await stt._on_transcript(
        {"type": "transcript", "text": "tomorrow evening. Actually, four", "is_final": False}
    )
    assert pushed == ["Actually, four"]


async def test_words_vad_missed_are_finalized_after_a_second():
    """VAD can miss a short word: then nothing asks for the final and the turn
    the word started would never end. The server sends an interim once, so the
    service asks for the final itself a second after the last new words."""
    stt = AirCapsLiveSTTService(api_key=KEY)
    pushed, sent = [], []

    async def push(frame, direction=None):
        pushed.append(frame.text)

    async def send(message):
        sent.append(message["type"])

    stt.push_frame = push
    stt._send_json = send
    stt._is_open = lambda: True
    stt.create_task = lambda coro, name=None: asyncio.ensure_future(coro)

    async def cancel_task(task):
        task.cancel()

    stt.cancel_task = cancel_task
    event = {"type": "transcript", "text": "loops", "is_final": False}
    await stt._on_transcript(event)
    await stt._on_transcript(event)  # a repeat is not pushed again
    assert pushed == ["loops"] and sent == []
    await asyncio.sleep(1.2)
    assert sent == ["finalize"]

    # while VAD hears the user, a pause in the words is not a reason to finalize
    stt._user_speaking = True
    sent.clear()
    await stt._on_transcript({"type": "transcript", "text": "loops again", "is_final": False})
    await asyncio.sleep(1.2)
    assert sent == []


async def test_end_of_turn_after_pipecat_ends_a_turn_itself():
    """2026-11-17: a final that did not end the turn ("Oh, I see", reason max_utterance) left the turn open; Pipecat
    ended it with its stop timeout. The next reply ("Yeah") must open a new turn, or Pipecat drops it."""
    async with FakeAirCaps([("Oh, I see.", "max_utterance"), "Yeah."]) as fake:
        stt = AirCapsLiveSTTService(api_key=KEY, base_url=fake.base_url, end_of_turn=True)
        # (0.2 s: the fake's turn length, so the next turn's words come only after Pipecat ends this one)
        speech = [*_speech(seconds=0.2), SleepFrame(0.3)]
        pipeline, timeline = _with_turns(stt)
        await run_test(
            pipeline, frames_to_send=[*speech, UserStoppedSpeakingFrame(), SleepFrame(0.1), *speech]
        )

    assert _turn_events(timeline) == [
        ("ProposedUserStartedSpeakingFrame", None),
        ("InterimTranscriptionFrame", "Oh,"),
        ("TranscriptionFrame", "Oh, I see."),
        ("ProposedUserStartedSpeakingFrame", None),
        ("InterimTranscriptionFrame", "Yeah."),
        ("TranscriptionFrame", "Yeah."),
        ("ProposedUserStoppedSpeakingFrame", None),
    ]


async def test_end_of_turn_words_follow_the_turn_start():
    """2026-11-18 demo: a new turn's first words were pushed before Pipecat had started the turn, so Pipecat's web
    client showed them at the end of the previous turn too. They now wait for the turn start."""
    async with FakeAirCaps(["Hi there.", "Book a table for two."]) as fake:
        stt = AirCapsLiveSTTService(api_key=KEY, base_url=fake.base_url, end_of_turn=True)
        # (Pipecat takes a while to start each turn)
        pipeline, timeline = _with_turns(stt, delay=0.2)
        speech = [*_speech(seconds=0.3), SleepFrame(0.6)]
        await run_test(pipeline, frames_to_send=[*speech, *speech])

    names = [e if isinstance(e, str) else type(e).__name__ for e in timeline]
    turn = [
        "ProposedUserStartedSpeakingFrame",
        "started",
        "InterimTranscriptionFrame",
        "TranscriptionFrame",
        "ProposedUserStoppedSpeakingFrame",
    ]
    assert names == turn * 2, names
    assert [e[1] for e in _turn_events(timeline) if e[1]] == [
        "Hi",
        "Hi there.",
        "Book a",
        "Book a table for two.",
    ]
