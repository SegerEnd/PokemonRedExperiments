import asyncio
import threading
import queue
import json

import gymnasium as gym

try:
    import websockets
except ImportError:
    websockets = None

X_POS_ADDRESS, Y_POS_ADDRESS = 0xD362, 0xD361
MAP_N_ADDRESS = 0xD35E

class StreamWrapper(gym.Wrapper):
    def __init__(self, env, stream_metadata={}):
        super().__init__(env)
        self.ws_address = "ws://localhost:8443/broadcast"
        self.stream_metadata = stream_metadata
        self.upload_interval = 30
        self.steam_step_counter = 0
        self.env = env
        self.coord_list = []
        if hasattr(env, "pyboy"):
            self.emulator = env.pyboy
        elif hasattr(env, "game"):
            self.emulator = env.game
        else:
            raise Exception("Could not find emulator!")

        # Non-blocking send queue + background thread
        self._send_queue = queue.Queue(maxsize=200)
        if websockets is not None:
            self._ws_thread = threading.Thread(target=self._send_loop, daemon=True)
            self._ws_thread.start()

    def _send_loop(self):
        loop = asyncio.new_event_loop()
        ws = None
        while True:
            msg = self._send_queue.get()
            try:
                if ws is None:
                    ws = loop.run_until_complete(
                        websockets.connect(self.ws_address)
                    )
                loop.run_until_complete(ws.send(msg))
            except Exception:
                ws = None
                # drain stale messages so we send fresh data on reconnect
                while not self._send_queue.empty():
                    try:
                        self._send_queue.get_nowait()
                    except queue.Empty:
                        break

    def step(self, action):
        result = self.env.step(action)

        x_pos = self.emulator.memory[X_POS_ADDRESS]
        y_pos = self.emulator.memory[Y_POS_ADDRESS]
        map_n = self.emulator.memory[MAP_N_ADDRESS]
        self.coord_list.append([x_pos, y_pos, map_n])

        if self.steam_step_counter >= self.upload_interval:
            self.stream_metadata["extra"] = f"coords: {len(self.env.seen_coords)}"
            msg = json.dumps({
                "metadata": self.stream_metadata,
                "coords": self.coord_list,
            })
            try:
                self._send_queue.put_nowait(msg)
            except queue.Full:
                pass  # drop message if behind
            self.steam_step_counter = 0
            self.coord_list = []

        self.steam_step_counter += 1
        return result
