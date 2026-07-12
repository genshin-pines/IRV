from backend.routers.alerts import router as alerts_router, ws_manager
from backend.routers.auth import router as auth_router
from backend.routers.cameras import router as cameras_router
from backend.routers.custom_gestures import router as custom_gestures_router
from backend.routers.gesture import router as gesture_router
from backend.routers.music import router as music_router
from backend.routers.plate import router as plate_router
from backend.routers.preferences import router as preferences_router
from backend.routers.traffic_police import router as traffic_police_router

__all__ = ["alerts_router", "auth_router", "ws_manager", "cameras_router", "custom_gestures_router", "gesture_router", "music_router", "plate_router", "preferences_router", "traffic_police_router"]
