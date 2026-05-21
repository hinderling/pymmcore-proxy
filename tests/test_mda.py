"""Tests for server-driven MDA execution."""

import threading
import time
from queue import Queue

import numpy as np
import pytest
from useq import MDAEvent, MDASequence


class TestMDABasics:
    def test_run_simple_sequence(self, core):
        """Run a simple time-lapse and collect frames."""
        frames = []

        def on_frame(img, event, meta):
            frames.append((img, event, meta))

        core.mda.events.frameReady.connect(on_frame)
        try:
            seq = MDASequence(
                time_plan={"interval": 0.01, "loops": 3},
            )
            core.mda.run(seq)
            assert len(frames) == 3
            for img, event, meta in frames:
                assert isinstance(img, np.ndarray)
                assert img.ndim == 2
        finally:
            core.mda.events.frameReady.disconnect(on_frame)

    def test_sequence_started_finished_signals(self, core):
        """sequenceStarted and sequenceFinished should fire."""
        started = []
        finished = []

        core.mda.events.sequenceStarted.connect(
            lambda seq, meta: started.append(seq)
        )
        core.mda.events.sequenceFinished.connect(
            lambda seq: finished.append(seq)
        )

        seq = MDASequence(time_plan={"interval": 0.01, "loops": 2})
        core.mda.run(seq)

        assert len(started) == 1
        assert len(finished) == 1


class TestMDAWithPositions:
    def test_z_stack(self, core):
        """Run a Z-stack and verify frames are collected."""
        frames = []
        core.mda.events.frameReady.connect(
            lambda img, ev, meta: frames.append(img)
        )

        seq = MDASequence(z_plan={"range": 4.0, "step": 2.0})
        core.mda.run(seq)

        # range=4, step=2 → 3 steps (-2, 0, 2)
        assert len(frames) >= 2


class TestMDACancel:
    def test_cancel_stops_sequence(self, core):
        """Canceling mid-sequence should stop and emit sequenceCanceled."""
        frames = []
        canceled = []

        def on_frame(img, ev, meta):
            frames.append(img)
            if len(frames) == 2:
                core.mda.cancel()

        core.mda.events.frameReady.connect(on_frame)
        core.mda.events.sequenceCanceled.connect(
            lambda seq: canceled.append(seq)
        )

        seq = MDASequence(time_plan={"interval": 0.01, "loops": 100})
        core.mda.run(seq)

        assert len(frames) < 100, "Sequence was not canceled"
        assert len(canceled) == 1

    def test_pause_and_resume(self, core):
        """Pausing should delay frames, resuming should continue."""
        frames = []
        paused_signals = []

        def on_frame(img, ev, meta):
            frames.append(img)
            if len(frames) == 1:
                core.mda.set_paused(True)
                # Resume after a short delay (from another thread)
                def resume():
                    time.sleep(0.3)
                    core.mda.set_paused(False)
                threading.Thread(target=resume, daemon=True).start()

        core.mda.events.frameReady.connect(on_frame)
        core.mda.events.sequencePauseToggled.connect(
            lambda p: paused_signals.append(p)
        )

        seq = MDASequence(time_plan={"interval": 0.01, "loops": 5})
        core.mda.run(seq)

        assert len(frames) == 5, "All frames should complete after resume"
        assert len(paused_signals) == 2  # pause + unpause


class TestMDAFromDict:
    def test_run_from_dict(self, core):
        """MDA.run() should accept a dict that gets converted to MDASequence."""
        frames = []
        core.mda.events.frameReady.connect(
            lambda img, ev, meta: frames.append(img)
        )

        core.mda.run({"time_plan": {"interval": 0.01, "loops": 2}})
        assert len(frames) == 2


class TestMDAStreaming:
    """Tests for streaming MDA with arbitrary iterables."""

    def test_run_from_list_of_events(self, core):
        """mda.run() should accept a list of MDAEvents (streamed)."""
        frames = []
        core.mda.events.frameReady.connect(
            lambda img, ev, meta: frames.append(img)
        )

        events = [MDAEvent(), MDAEvent(), MDAEvent()]
        core.mda.run(events)

        assert len(frames) == 3
        for img in frames:
            assert isinstance(img, np.ndarray)

    def test_run_from_generator(self, core):
        """mda.run() should accept a generator of MDAEvents."""
        frames = []
        core.mda.events.frameReady.connect(
            lambda img, ev, meta: frames.append(img)
        )

        def event_gen():
            for _ in range(4):
                yield MDAEvent()

        core.mda.run(event_gen())
        assert len(frames) == 4

    def test_run_from_queue(self, core):
        """mda.run() should accept a Queue-backed iterable."""
        frames = []
        finished = []

        core.mda.events.frameReady.connect(
            lambda img, ev, meta: frames.append(img)
        )
        core.mda.events.sequenceFinished.connect(
            lambda seq: finished.append(seq)
        )

        q: Queue = Queue()
        _sentinel = object()

        class QueueIterator:
            def __iter__(self):
                return self

            def __next__(self):
                item = q.get()
                if item is _sentinel:
                    raise StopIteration
                return item

        # Pre-fill three events, then sentinel
        for _ in range(3):
            q.put(MDAEvent())
        q.put(_sentinel)

        core.mda.run(QueueIterator())

        assert len(frames) == 3
        assert len(finished) == 1

    def test_queue_reactive_feedback(self, core):
        """Events can be added to a queue in response to frameReady signals.

        This is the key use-case: a feedback loop where each acquired frame
        informs the next event to execute.
        """
        frames = []
        q: Queue = Queue()
        _sentinel = object()

        class QueueIterator:
            def __iter__(self):
                return self

            def __next__(self):
                item = q.get()
                if item is _sentinel:
                    raise StopIteration
                return item

        def on_frame(img, ev, meta):
            frames.append(img)
            if len(frames) < 3:
                # React to the acquired frame by adding another event
                q.put(MDAEvent())
            else:
                # Done — stop the sequence
                q.put(_sentinel)

        core.mda.events.frameReady.connect(on_frame)

        # Seed with the first event
        q.put(MDAEvent())
        core.mda.run(QueueIterator())

        assert len(frames) == 3

    def test_streaming_signals(self, core):
        """sequenceStarted and sequenceFinished should fire for streamed MDA."""
        started = []
        finished = []

        core.mda.events.sequenceStarted.connect(
            lambda seq, meta: started.append(seq)
        )
        core.mda.events.sequenceFinished.connect(
            lambda seq: finished.append(seq)
        )

        events = [MDAEvent(), MDAEvent()]
        core.mda.run(events)

        assert len(started) == 1
        assert len(finished) == 1

    def test_streaming_cancel(self, core):
        """Canceling a streaming MDA should stop execution."""
        frames = []
        canceled = []

        def on_frame(img, ev, meta):
            frames.append(img)
            if len(frames) == 2:
                core.mda.cancel()

        core.mda.events.frameReady.connect(on_frame)
        core.mda.events.sequenceCanceled.connect(
            lambda seq: canceled.append(seq)
        )

        # Generate many events — cancel should stop early
        events = [MDAEvent() for _ in range(100)]
        core.mda.run(events)

        assert len(frames) < 100
        assert len(canceled) == 1
