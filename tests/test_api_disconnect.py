"""A client that disconnects mid-request is one quiet line, not a traceback; real errors still use the default path."""
import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from penumbra.api import QuietServer


class QuietServerTest(unittest.TestCase):
    def call(self, error):
        server = QuietServer.__new__(QuietServer)  # no socket needed for handle_error
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), mock.patch.object(sys, "exc_info", return_value=(type(error), error, None)):
            with mock.patch("socketserver.BaseServer.handle_error") as default:
                server.handle_error(None, ("127.0.0.1", 50123))
        return out.getvalue(), default.called

    def test_client_gone_is_one_line(self):
        for error in (ConnectionResetError(10054, "reset"), BrokenPipeError(), ConnectionAbortedError()):
            out, default_called = self.call(error)
            self.assertIn("disconnected", out)
            self.assertFalse(default_called)

    def test_real_errors_keep_the_traceback(self):
        out, default_called = self.call(RuntimeError("boom"))
        self.assertTrue(default_called)
        self.assertEqual(out, "")


if __name__ == "__main__":
    unittest.main()
