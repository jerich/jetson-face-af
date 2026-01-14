"""Threaded Nikon camera controller wrapping NikonMCN10Emu."""

import threading
import queue
import time
import logging
import sys
import os

import config
from pipeline.command_logic import Command

logger = logging.getLogger(__name__)


class NikonAFController:
    """Threaded wrapper around NikonMCN10Emu for AF steering commands.

    Runs a background thread that:
    - Sends heartbeat every HEARTBEAT_INTERVAL_SEC when idle
    - Processes commands from a queue as fast as they arrive
    - Stops commanding on USB error (no auto-reconnect)
    """

    def __init__(self, dry_run: bool = False):
        self._dry_run = dry_run
        self._emu = None
        self._command_queue: queue.Queue[Command] = queue.Queue()
        self._running = False
        self._thread = None
        self._connected = False
        self._lock = threading.Lock()

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self):
        """Connect to camera and start the controller thread."""
        if self._dry_run:
            logger.info("Nikon controller starting in DRY RUN mode")
            self._connected = True
        else:
            self._connect()

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _connect(self):
        """Initialize connection to the Nikon camera."""
        try:
            # Add nikon-usb-control directory to path for direct import
            nikon_path = os.path.expanduser("~/nikon-usb-control/nikon-usb-control")
            if nikon_path not in sys.path:
                sys.path.insert(0, nikon_path)
            import Nikon_mc_n10
            # Reduce drain timeout for faster command throughput
            Nikon_mc_n10.DRAIN_OVERALL_TIME_MS = 300

            self._emu = Nikon_mc_n10.NikonMCN10Emu(verbose=False)
            logger.info("Connecting to Nikon camera...")
            self._emu.connect_and_init()
            self._connected = True
            logger.info("Nikon camera connected")
        except Exception as e:
            logger.error(f"Failed to connect to Nikon camera: {e}")
            self._connected = False
            raise

    def send_command(self, command: Command):
        """Queue a command for the controller thread to execute."""
        if command != Command.NONE:
            self._command_queue.put(command)

    def stop(self):
        """Stop the controller thread and disconnect."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._emu is not None:
            try:
                self._emu.close()
            except Exception as e:
                logger.warning(f"Error closing Nikon connection: {e}")
            self._emu = None
        self._connected = False

    def _run_loop(self):
        """Main controller loop: process commands or send heartbeat."""
        last_heartbeat = 0.0

        while self._running:
            if not self._connected:
                time.sleep(0.1)
                continue

            # Process any pending commands
            try:
                command = self._command_queue.get(timeout=0.05)
                self._execute_command(command)
                last_heartbeat = time.time()
                continue
            except queue.Empty:
                pass

            # Send heartbeat if enough time has elapsed
            now = time.time()
            if now - last_heartbeat >= config.HEARTBEAT_INTERVAL_SEC:
                self._send_heartbeat()
                last_heartbeat = now

    def _execute_command(self, command: Command):
        """Execute a single command on the camera."""
        if self._dry_run:
            logger.info(f"DRY RUN: would send {command.value}")
            return

        try:
            if command == Command.LEFT:
                self._emu.multi_left()
            elif command == Command.RIGHT:
                self._emu.multi_right()
            elif command == Command.FN1:
                self._emu.press_fn1()
            logger.debug(f"Sent command: {command.value}")
        except Exception as e:
            logger.error(f"USB error sending command {command.value}: {e}")
            self._connected = False

    def _send_heartbeat(self):
        """Send an idle heartbeat to keep the session alive."""
        if self._dry_run:
            return

        try:
            self._emu.send_idle_heartbeat()
        except Exception as e:
            logger.error(f"USB error sending heartbeat: {e}")
            self._connected = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
