"""Integration tests for RPC calls through the proxy."""

import httpx
import numpy as np
import pytest


class TestHealthAndBasics:
    def test_health(self, core_no_signals):
        result = core_no_signals.health()
        assert result["status"] == "ok"

    def test_repr(self, core_no_signals):
        assert "RemoteMMCore" in repr(core_no_signals)


class TestInfoEndpoint:
    def test_info_returns_200(self, server_url):
        r = httpx.get(f"{server_url}/info", timeout=3.0)
        assert r.status_code == 200

    def test_info_has_core_type_key(self, server_url):
        data = httpx.get(f"{server_url}/info", timeout=3.0).json()
        assert "core_type" in data

    def test_info_core_type_is_cmmcoreplus(self, server_url):
        """The conftest demo_core fixture creates a CMMCorePlus instance."""
        data = httpx.get(f"{server_url}/info", timeout=3.0).json()
        assert data["core_type"] == "CMMCorePlus"

    def test_info_does_not_break_health(self, server_url):
        """Calling /info must not affect /health."""
        httpx.get(f"{server_url}/info", timeout=3.0)
        r = httpx.get(f"{server_url}/health", timeout=3.0)
        assert r.json()["status"] == "ok"

    def test_info_core_type_unicore(self):
        """A server backed by UniMMCore returns 'UniMMCore'."""
        import socket
        import threading
        import time
        import uvicorn
        from pymmcore_plus.experimental.unicore import UniMMCore
        from pymmcore_proxy import ProxyServer

        core = UniMMCore()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        proxy = ProxyServer(core, port=port)
        config = uvicorn.Config(proxy.app, host="127.0.0.1", port=port, log_level="warning", ws="wsproto")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{url}/health", timeout=1.0).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.1)
        else:
            server.should_exit = True
            pytest.fail("UniMMCore proxy server did not start in time")

        try:
            data = httpx.get(f"{url}/info", timeout=3.0).json()
            assert data["core_type"] == "UniMMCore"
        finally:
            server.should_exit = True
            thread.join(timeout=5.0)


class TestDeviceInfo:
    def test_get_loaded_devices(self, core_no_signals):
        devices = core_no_signals.getLoadedDevices()
        assert isinstance(devices, (list, tuple))
        assert len(devices) > 0

    def test_get_camera_device(self, core_no_signals):
        cam = core_no_signals.getCameraDevice()
        assert isinstance(cam, str)
        assert len(cam) > 0

    def test_get_xy_stage_device(self, core_no_signals):
        stage = core_no_signals.getXYStageDevice()
        assert isinstance(stage, str)


class TestProperties:
    def test_get_exposure(self, core_no_signals):
        exp = core_no_signals.getExposure()
        assert isinstance(exp, (int, float))
        assert exp > 0

    def test_set_and_get_exposure(self, core_no_signals):
        core_no_signals.setExposure(100.0)
        assert core_no_signals.getExposure() == pytest.approx(100.0)
        core_no_signals.setExposure(50.0)  # restore

    def test_get_image_width_height(self, core_no_signals):
        w = core_no_signals.getImageWidth()
        h = core_no_signals.getImageHeight()
        assert isinstance(w, int)
        assert isinstance(h, int)
        assert w > 0
        assert h > 0


class TestImaging:
    def test_snap_and_get_image(self, core_no_signals):
        core_no_signals.snapImage()
        img = core_no_signals.getImage()
        assert isinstance(img, np.ndarray)
        assert img.ndim == 2
        assert img.shape[0] > 0
        assert img.shape[1] > 0

    def test_image_is_writable(self, core_no_signals):
        core_no_signals.snapImage()
        img = core_no_signals.getImage()
        img[0, 0] = 0  # should not raise


class TestStage:
    def test_set_and_get_xy(self, core_no_signals):
        core_no_signals.setXYPosition(123.0, 456.0)
        core_no_signals.waitForDevice(core_no_signals.getXYStageDevice())
        x = core_no_signals.getXPosition()
        y = core_no_signals.getYPosition()
        assert x == pytest.approx(123.0, abs=1.0)
        assert y == pytest.approx(456.0, abs=1.0)

    def test_set_and_get_z(self, core_no_signals):
        focus = core_no_signals.getFocusDevice()
        core_no_signals.setPosition(focus, 10.0)
        core_no_signals.waitForDevice(focus)
        z = core_no_signals.getPosition(focus)
        assert z == pytest.approx(10.0, abs=1.0)


class TestConfig:
    def test_get_available_config_groups(self, core_no_signals):
        groups = core_no_signals.getAvailableConfigGroups()
        assert isinstance(groups, (list, tuple))

    def test_get_available_configs(self, core_no_signals):
        groups = core_no_signals.getAvailableConfigGroups()
        if groups:
            configs = core_no_signals.getAvailableConfigs(groups[0])
            assert isinstance(configs, (list, tuple))


class TestErrorHandling:
    def test_nonexistent_method(self, core_no_signals):
        with pytest.raises(AttributeError):
            core_no_signals.thisMethodDoesNotExist()

    def test_bad_args(self, core_no_signals):
        with pytest.raises(TypeError):
            core_no_signals.setExposure("not_a_number")
