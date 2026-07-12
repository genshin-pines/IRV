# app.py - FastAPI backend
import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse

from stream_manager import StreamManager

# ========== Config (edit here only) ==========
HOST = "0.0.0.0"
PORT = 8000
DEFAULT_SRC_URL = "rtsp://10.126.59.120:8554/live/live1"
# =============================================

app = FastAPI(title="Gesture Vehicle Control")

manager = None
connected_clients = set()
frontend_dir = Path(__file__).parent.parent / "frontend"


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(frontend_dir / "index.html")


@app.get("/video_feed")
async def video_feed():
    import cv2

    async def generate():
        while True:
            if manager and manager.is_running:
                frame = manager.get_latest_frame()
                if frame is not None:
                    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
            await asyncio.sleep(0.04)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.websocket("/ws")
async def gesture_websocket(ws: WebSocket):
    global manager
    await ws.accept()
    connected_clients.add(ws)

    async def broadcast(msg):
        for client in list(connected_clients):
            try:
                await client.send_json(msg)
            except Exception:
                connected_clients.discard(client)

    queue_task = None

    async def consume_queue():
        while manager and manager.is_running:
            try:
                msg_type, data = manager.out_queue.get(timeout=0.05)
                await broadcast(data)
            except Exception:
                pass
            await asyncio.sleep(0.01)

    try:
        while True:
            msg = await ws.receive_json()
            cmd = msg.get("cmd", "")

            if cmd == "start":
                use_webcam = msg.get("use_webcam", False)
                src_url = msg.get("src_url", DEFAULT_SRC_URL) if not use_webcam else None

                if manager and manager.is_running:
                    await asyncio.to_thread(manager.stop)

                manager = StreamManager(src_url=src_url, use_webcam=use_webcam)
                await asyncio.to_thread(manager.start)

                if manager.is_running:
                    await ws.send_json({
                        "type": "status", "status": "started",
                        "hls_url": manager.hls_url,
                        "mjpg_url": "/video_feed",
                        "rtsp_url": manager.dst_url,
                    })
                    if queue_task:
                        queue_task.cancel()
                    queue_task = asyncio.create_task(consume_queue())
                else:
                    await ws.send_json({"type": "error", "message": manager.error or "Cannot open source"})
                    manager = None

            elif cmd == "stop":
                if queue_task:
                    queue_task.cancel()
                    queue_task = None
                if manager:
                    await asyncio.to_thread(manager.stop)
                    manager = None
                await ws.send_json({"type": "status", "status": "stopped"})

            elif cmd == "ping":
                await ws.send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass
    finally:
        connected_clients.discard(ws)
        if queue_task:
            queue_task.cancel()
        if len(connected_clients) == 0 and manager:
            await asyncio.to_thread(manager.stop)
            manager = None


if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("  Gesture Vehicle Control System")
    print(f"  Frontend: http://127.0.0.1:{PORT}")
    print(f"  MJPG:     http://127.0.0.1:{PORT}/video_feed")
    print(f"  WS:       ws://127.0.0.1:{PORT}/ws")
    print("=" * 50)
    uvicorn.run(app, host=HOST, port=PORT)