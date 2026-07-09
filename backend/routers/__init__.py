from backend.routers.alerts import router as alerts_router, ws_manager
from backend.routers.cameras import router as cameras_router
from backend.routers.gesture import router as gesture_router
from backend.routers.plate import router as plate_router

__all__ = ["alerts_router", "ws_manager", "cameras_router", "gesture_router", "plate_router"]
