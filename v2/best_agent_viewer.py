import asyncio
import base64
import io
import json
import queue
import threading
import numpy as np
from PIL import Image
from stable_baselines3.common.callbacks import BaseCallback

try:
    import websockets
except ImportError:
    websockets = None


class BestAgentViewer(BaseCallback):

    def __init__(self, update_interval=12, ws_address="ws://dev.segerend.nl:8443/broadcast",
                 user="anonymous", color="#ffffff", verbose=0):
        super().__init__(verbose)
        self.update_interval = update_interval
        self.ws_address = ws_address
        self.user = user
        self.color = color
        self._send_queue = queue.Queue(maxsize=100)
        self._watched = set()
        self._watch_best = True
        self._lock = threading.Lock()
        self._thread = None
        self._running = False

    def _on_training_start(self):
        if websockets is None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._ws_loop, daemon=True)
        self._thread.start()

    def _ws_loop(self):
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self._run())

    async def _run(self):
        while self._running:
            try:
                async with websockets.connect(self.ws_address) as ws:
                    await asyncio.gather(
                        self._sender(ws),
                        self._listener(ws),
                    )
            except Exception:
                await asyncio.sleep(2)

    async def _sender(self, ws):
        """Send queued messages."""
        while self._running:
            try:
                msg = self._send_queue.get_nowait()
                await ws.send(msg)
            except queue.Empty:
                await asyncio.sleep(0.01)
            except Exception:
                return

    async def _listener(self, ws):
        """Listen for watch/unwatch requests."""
        while self._running:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                data = json.loads(raw)
                if data.get("user") and data["user"] != self.user:
                    continue
                with self._lock:
                    if data.get("type") == "watch":
                        aid = data.get("agent_id")
                        if aid is not None:
                            self._watched.add(int(aid))
                            self._watch_best = False
                            print(f"[BestAgentViewer] Watching agent {aid}, watch_best={self._watch_best}")
                    elif data.get("type") == "unwatch":
                        aid = data.get("agent_id")
                        if aid is not None:
                            self._watched.discard(int(aid))
                    elif data.get("type") == "watch_best":
                        self._watch_best = True
                        self._watched.clear()
            except asyncio.TimeoutError:
                continue
            except Exception:
                return

    def _encode_frame(self, frame):
        buf = io.BytesIO()
        Image.fromarray(frame, mode="L").save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _on_step(self) -> bool:
        if not self._running:
            return True
        if self.n_calls % self.update_interval != 0:
            return True
        try:
            rewards = self.training_env.get_attr("total_reward")
            deaths = self.training_env.get_attr("died_count")
            resets = self.training_env.get_attr("reset_count")
            best_idx = int(np.argmax(rewards))

            roster = json.dumps({
                "type": "roster",
                "user": self.user,
                "color": self.color,
                "best_idx": best_idx,
                "total_births": sum(resets),
                "total_deaths": sum(deaths),
                "agents": [
                    {"id": i, "reward": float(r), "deaths": d, "resets": rc}
                    for i, (r, d, rc) in enumerate(zip(rewards, deaths, resets))
                ],
            })

            with self._lock:
                watch_best = self._watch_best
                watched = self._watched.copy()

            screen_indices = set(watched)
            if watch_best:
                screen_indices.add(best_idx)

            try:
                self._send_queue.put_nowait(roster)
            except queue.Full:
                pass

            for idx in screen_indices:
                if 0 <= idx < len(rewards):
                    frame = self.training_env.env_method("render_full", indices=[idx])[0]
                    msg = json.dumps({
                        "type": "screen",
                        "user": self.user,
                        "color": self.color,
                        "agent_id": idx,
                        "reward": float(rewards[idx]),
                        "is_best": idx == best_idx,
                        "image": self._encode_frame(frame),
                    })
                    try:
                        self._send_queue.put_nowait(msg)
                    except queue.Full:
                        pass

        except Exception as e:
            if self.verbose > 0:
                print(f"BestAgentViewer error: {e}")
        return True

    def _on_training_end(self):
        self._running = False
