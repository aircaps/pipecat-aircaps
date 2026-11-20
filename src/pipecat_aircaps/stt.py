#
# Copyright (c) 2026, AirCaps
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AirCaps Live 1 speech-to-text service for Pipecat.

AirCaps Live 1 (``aircaps-live-1``) is a low-latency English streaming
speech-to-text model for voice agents. Audio streams over a WebSocket and the
transcript comes back as the user speaks. Turns end one of two ways: Pipecat's
VAD reports that the user stopped speaking and the service asks for the final
transcript at once, or, with ``end_of_turn=True``, the model detects the end of
the turn itself.
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Literal

import aiohttp
from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    StartFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.types import NOT_GIVEN, NotGiven, is_given
from websockets.protocol import State

# P99 time from end of speech to final transcript, with VADParams.stop_secs=0.2
# (the Pipecat default), measured from US East. Includes the VAD stop wait.
AIRCAPS_LIVE_TTFS_P99 = 0.25

# The model takes 16 kHz mono PCM16; other input rates are resampled to it.
_SAMPLE_RATE = 16000

# The server closes a session that sends nothing for 45 seconds (for example
# while the STT is muted), so an idle connection is pinged before then.
_KEEPALIVE_TIMEOUT = 20.0

# Punctuation that can close the previous turn's text after it was final.
_LEADING_PUNCTUATION = " .,!?;:"

# End-of-turn settings that override single values of the mode.
_END_OF_TURN_OVERRIDES = (
    "end_of_turn_threshold",
    "min_turn_silence_ms",
    "max_turn_silence_ms",
    "vad_threshold",
)

# New words that VAD did not hear are finalized after this many seconds without
# newer words (see _restart_missed_speech_timer).
_MISSED_SPEECH_WAIT = 1.0
# With end_of_turn: how long a new turn's first transcript waits for Pipecat to
# start the turn it proposed (see _on_transcript).
_TURN_START_WAIT = 0.5

# How long to wait for the session to be ready after connecting.
_READY_TIMEOUT = 15.0


class _LookupError(ConnectionError):
    """The endpoint lookup was refused; ``status`` lets Pipecat classify why."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass
class AirCapsLiveSTTSettings(STTSettings):
    """Settings for AirCapsLiveSTTService.

    Parameters:
        model: The AirCaps AI model id. Defaults to ``"aircaps-live-1"``.
        language: Transcription language. AirCaps Live 1 transcribes English.
        end_of_turn_mode: How the model trades speed for certainty when it
            detects the end of a turn: ``"min_latency"``, ``"balanced"`` (the
            default) or ``"max_accuracy"``. Used only with ``end_of_turn=True``.
            The fields below override single values of the mode; None keeps
            the mode's value.
        end_of_turn_threshold: How sure the model must be that the turn ended,
            0.2 to 0.9 (modes: 0.3, 0.5, 0.6).
        min_turn_silence_ms: The shortest pause that can end a turn, 0 to 1000
            ms (modes: 40, 80, 400).
        max_turn_silence_ms: A pause this long always ends the turn: 500 to
            5000 ms, or 0 for no limit (modes: 2560).
        vad_threshold: How sure the model's voice detector must be that it
            hears speech, 0.3 to 0.9.
    """

    end_of_turn_mode: Literal["min_latency", "balanced", "max_accuracy"] | NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )
    end_of_turn_threshold: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    min_turn_silence_ms: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    max_turn_silence_ms: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    vad_threshold: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class AirCapsLiveSTTService(WebsocketSTTService):
    """Speech-to-text service using AirCaps Live 1.

    Streams audio to AirCaps AI over a WebSocket and pushes interim transcripts as
    the user speaks.

    By default Pipecat decides when a turn ends: when VAD reports that the user
    stopped speaking, the service sends ``finalize`` and pushes the turn's final
    ``TranscriptionFrame`` as soon as it arrives.

    With ``end_of_turn=True`` the model decides instead. The service pushes a
    ``ProposedUserStartedSpeakingFrame`` when a turn's first words arrive, and
    the turn's final ``TranscriptionFrame`` followed by a
    ``ProposedUserStoppedSpeakingFrame`` when the model ends the turn. It
    recommends ``ExternalUserTurnStrategies``, so the user aggregator follows
    these frames rather than its own VAD and turn analyzer.

    Before each connection the service asks the AirCaps AI API where to connect
    (``GET /v1/realtime/endpoint``), so audio goes straight to the machine
    running the model.
    """

    Settings = AirCapsLiveSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.aircaps.com",
        sample_rate: int | None = None,
        end_of_turn: bool = False,
        should_interrupt: bool = True,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = AIRCAPS_LIVE_TTFS_P99,
        **kwargs,
    ):
        """Initialize the AirCaps Live 1 STT service.

        Args:
            api_key: AirCaps AI API key.
            base_url: AirCaps AI API address, used to look up the WebSocket address.
            sample_rate: Input audio sample rate. If None, uses the pipeline's
                input rate. Audio is resampled to 16 kHz before it is sent.
            end_of_turn: Whether the model detects the end of each turn, rather
                than Pipecat's VAD and turn strategies.
            should_interrupt: With ``end_of_turn``, whether a new user turn
                interrupts the bot. Passed to the recommended
                ``ExternalUserTurnStrategies``; a user-supplied
                ``user_turn_strategies`` overrides it.
            settings: Runtime-updatable settings.
            ttfs_p99_latency: P99 latency from speech end to final transcript in
                seconds. Override for your deployment. See
                https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to the parent WebsocketSTTService.
        """
        default_settings = self.Settings(
            model="aircaps-live-1",
            language=Language.EN,
            end_of_turn_mode="balanced",
            end_of_turn_threshold=None,
            min_turn_silence_ms=None,
            max_turn_silence_ms=None,
            vad_threshold=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            keepalive_timeout=_KEEPALIVE_TIMEOUT,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._end_of_turn = end_of_turn
        self._should_interrupt = should_interrupt
        # With end_of_turn: whether a turn has started and not yet ended, and
        # whether Pipecat has started the user turn proposed for it.
        self._in_turn = False
        self._turn_started = asyncio.Event()
        self._receive_task: asyncio.Task | None = None
        # The session's whole transcript so far, and how much of it is final.
        self._text = ""
        self._final_length = 0
        # The last interim pushed: an unchanged repeat is not pushed again.
        self._last_interim = ""
        self._missed_speech_task: asyncio.Task | None = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as AirCaps Live 1 STT supports metrics generation.
        """
        return True

    @property
    def send_sample_rate(self) -> int:
        """Get the rate audio is sent at: always 16 kHz.

        Returns:
            The sample rate in Hz.
        """
        return _SAMPLE_RATE

    @property
    def supports_ttfs(self) -> bool:
        """Whether a speech-end to final-transcript time applies.

        Returns:
            False with ``end_of_turn``, where the model defines the turn's end.
        """
        return not self._end_of_turn

    def service_metadata_frame(self) -> STTMetadataFrame:
        """Recommend turn strategies that follow the model's turns, with ``end_of_turn``.

        Returns:
            The STT metadata frame broadcast at start.
        """
        frame = super().service_metadata_frame()
        if self._end_of_turn:
            frame.user_turn_strategies = ExternalUserTurnStrategies(
                enable_interruptions=self._should_interrupt
            )
        return frame

    async def start(self, frame: StartFrame):
        """Start the service and connect.

        Args:
            frame: The start frame.
        """
        await super().start(frame)
        await self._connect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, asking for the final transcript when the user stops speaking.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._turn_started.set()
        if isinstance(frame, UserStoppedSpeakingFrame):
            # Pipecat ended the user turn, maybe on its own (its stop timeout, when
            # a final that does not end the turn is the last thing the model sent).
            # The next words start a new turn: propose its start, or Pipecat never
            # opens it and drops what the user says next.
            self._in_turn = False
        if isinstance(frame, VADUserStoppedSpeakingFrame) and not self._end_of_turn:
            await self._cancel_missed_speech_timer()
            await self._finalize()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Send audio to AirCaps.

        Args:
            audio: Raw 16-bit mono PCM audio at 16 kHz.

        Yields:
            None, as transcripts arrive on the WebSocket.
        """
        # The receive loop reconnects a dropped socket; audio arriving
        # meanwhile is dropped.
        if self._is_open():
            try:
                await self._websocket.send(audio)
            except Exception as e:
                logger.warning(f"{self}: send failed: {e}")
        yield None

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta, reconnecting to use it.

        Args:
            delta: An STT settings delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)
        if changed:
            await self._request_reconnect()
        return changed

    async def _connect(self):
        await self._connect_websocket()

        await super()._connect()

        # Started even when the connection failed: with no socket the receive
        # loop goes straight to its reconnect path, which retries the connect.
        if not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()
        await self._cancel_missed_speech_timer()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        if self._is_open():
            return
        try:
            logger.debug("Connecting to AirCaps Live 1 STT")
            url = await self._websocket_url()
            websocket = await self._websocket_connect(
                url, additional_headers={"Authorization": f"Bearer {self._api_key}"}
            )
            try:
                await self._start_session(websocket)
            except BaseException:
                await websocket.close()
                raise
            self._text = ""
            self._final_length = 0
            self._last_interim = ""
            self._websocket = websocket
            logger.info(f"{self}: Connected to AirCaps Live 1 STT")
            await self._call_event_handler("on_connected")
        except Exception as e:
            self._websocket = None
            await self.push_error(error_msg=f"Unable to connect to AirCaps AI: {e}", exception=e)

    async def _disconnect_websocket(self):
        websocket = self._websocket
        try:
            if websocket and websocket.state is State.OPEN:
                logger.debug(f"{self}: Disconnecting from AirCaps Live 1 STT")
                await websocket.send(json.dumps({"type": "finish"}))
                await websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            # Only clear if no concurrent connect has already replaced it.
            if self._websocket is websocket:
                self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def _websocket_url(self) -> str:
        """Ask the AirCaps AI API where to open this model's WebSocket."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self._base_url}/v1/realtime/endpoint",
                params={"model": str(self._settings.model)},
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                body = await response.json(content_type=None)
                if response.status != 200:
                    error = body.get("error") if isinstance(body, dict) else None
                    if not isinstance(error, dict):
                        error = {}
                    # A refused key or model is permanent: the service stops retrying.
                    raise _LookupError(
                        response.status,
                        f"endpoint lookup failed ({response.status} "
                        f"{error.get('code', 'error')}): {error.get('message', body)}",
                    )
                return body["websocket_url"]

    async def _start_session(self, websocket):
        """Send the start message and wait until the session is ready for audio."""
        start = {
            "type": "start",
            "model": self._settings.model,
            "encoding": "pcm_s16le",
            "sample_rate": _SAMPLE_RATE,
            "channels": 1,
        }
        if self._end_of_turn:
            start.update(end_of_turn=True, end_of_turn_mode=self._settings.end_of_turn_mode)
            for name in _END_OF_TURN_OVERRIDES:
                value = getattr(self._settings, name)
                if is_given(value) and value is not None:
                    start[name] = value
        await websocket.send(json.dumps(start))
        async with asyncio.timeout(_READY_TIMEOUT):
            async for message in websocket:
                data = json.loads(message)
                if data.get("type") == "ready":
                    return
                if data.get("type") == "error":
                    raise ConnectionError(f"{data.get('code')}: {data.get('message')}")
        raise ConnectionError("connection closed before the session was ready")

    def _restart_missed_speech_timer(self):
        """Ask for the final if new words arrive that VAD did not hear.

        VAD can miss a short or quiet word. Then it never reports the user
        stopping, nothing asks for the final, and the turn the words started
        stays open. If no newer words arrive within ``_MISSED_SPEECH_WAIT``
        while VAD hears no speech, the service asks for the final itself.
        """
        if self._missed_speech_task:
            self._missed_speech_task.cancel()
        self._missed_speech_task = self.create_task(self._finalize_missed_speech())

    async def _finalize_missed_speech(self):
        await asyncio.sleep(_MISSED_SPEECH_WAIT)
        self._missed_speech_task = None
        if not self._user_speaking:
            await self._finalize()

    async def _cancel_missed_speech_timer(self):
        if self._missed_speech_task:
            task, self._missed_speech_task = self._missed_speech_task, None
            await self.cancel_task(task)

    async def _finalize(self):
        """Ask for the final transcript of everything heard so far."""
        if self._is_open():
            self.request_finalize()
            await self._send_json({"type": "finalize"})

    def _is_open(self) -> bool:
        return self._websocket is not None and self._websocket.state is State.OPEN

    async def _send_json(self, message: dict):
        try:
            await self._websocket.send(json.dumps(message))
        except Exception as e:
            logger.warning(f"{self}: send failed: {e}")

    async def _send_keepalive(self, silence: bytes):
        """Ping the session so the server keeps an idle connection open.

        Args:
            silence: Unused: a ping keeps the session open without sending audio.
        """
        await self._send_json({"type": "ping"})

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        """Process incoming WebSocket messages."""
        try:
            async for message in self._get_websocket():
                try:
                    await self._process_message(json.loads(message))
                except json.JSONDecodeError:
                    logger.warning(f"{self}: received non-JSON message: {message}")
                except Exception as e:
                    logger.error(f"{self}: error processing message: {e}")
        finally:
            # A turn lives in one session, so it ends when the session does.
            if self._in_turn and not self._disconnecting:
                await self._end_turn(self._text[self._final_length :].strip(), None)

    async def _process_message(self, data: dict):
        kind = data.get("type")
        if kind == "transcript":
            await self._on_transcript(data)
        elif kind == "error":
            await self.push_error(
                error_msg=f"AirCaps AI error {data.get('code')}: {data.get('message')}"
            )
        elif kind == "session_limit":
            # The server ends the session next; the receive loop reconnects.
            logger.warning(f"{self}: {data.get('message')}")

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def _on_transcript(self, data: dict):
        # Each event carries the session's whole transcript; the characters
        # before the last final never change, so the current turn is the rest.
        self._text = data.get("text", "")
        # The model may punctuate the end of the previous turn once it hears
        # more ("evening" becomes "evening. Actually ..."): that punctuation
        # belongs to the previous turn, not this one.
        turn = self._text[self._final_length :].lstrip(_LEADING_PUNCTUATION).strip()
        language = Language.EN

        # Text that is only punctuation is not speech: the model can add "?" to
        # the end of the previous turn after its final. Pushed as a transcript,
        # it would start a new user turn in Pipecat and interrupt the bot.
        if not any(c.isalnum() for c in turn):
            turn = ""

        if self._end_of_turn and not self._in_turn and turn:
            self._in_turn = True
            self._turn_started.clear()
            await self.broadcast_frame(ProposedUserStartedSpeakingFrame)
            # The turn's words go out once Pipecat has started the turn: an app
            # that files transcripts under the user turn open when they arrive
            # (Pipecat's web client) would otherwise put them in the previous one.
            try:
                await asyncio.wait_for(self._turn_started.wait(), _TURN_START_WAIT)
            except TimeoutError:
                pass

        if not data.get("is_final"):
            if turn and turn != self._last_interim:
                self._last_interim = turn
                await self.push_frame(
                    InterimTranscriptionFrame(
                        turn, self._user_id, time_now_iso8601(), language, result=data
                    )
                )
                if not self._end_of_turn:
                    self._restart_missed_speech_timer()
            return

        self._final_length = len(self._text)
        self._last_interim = ""
        await self._cancel_missed_speech_timer()
        if self._end_of_turn and data.get("end_of_turn"):
            # The model ended the turn: its final, then the turn's end.
            if self._in_turn:
                await self._end_turn(turn, data)
            return

        # A final within a turn (end_of_turn), or the answer to finalize.
        self.confirm_finalize()
        await self._push_final(turn, data)

    async def _end_turn(self, turn: str, data: dict | None):
        """Push the turn's final transcript and propose the end of the turn.

        ``data`` is None for a turn cut short by its session ending: its text so
        far is pushed, not marked finalized, since the model never ended it.
        """
        self._in_turn = False
        await self._push_final(turn, data, finalized=data is not None)
        await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)

    async def _push_final(self, turn: str, data: dict | None, finalized: bool = False):
        if not turn:
            return
        # Report usage before the transcription frame so tracing can attach it
        # to the STT span the frame closes.
        await self.emit_stt_usage_metrics()
        await self.push_frame(
            TranscriptionFrame(
                turn,
                self._user_id,
                time_now_iso8601(),
                Language.EN,
                result=data,
                finalized=finalized,
            )
        )
        await self._handle_transcription(turn, True, Language.EN)
