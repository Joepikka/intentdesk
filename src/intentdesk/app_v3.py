"""Add a separate hybrid card review desk without modifying the bound original app."""

from pathlib import Path
from threading import Lock

from fastapi import HTTPException, Query
from fastapi.responses import FileResponse

from intentdesk.app import Decision, NewRequest, create_app as original_app
from intentdesk.card_v3 import CardRouter
from intentdesk.pilot import CATEGORIES, handoff
from intentdesk.reviews import ReviewConflict, ReviewStore


def create_app(root: Path, router_factory=None, database=None, card_factory=None):
    app = original_app(root, router_factory, database)
    app.state.card_router = None
    app.state.card_store = None
    init_lock = Lock()

    def store():
        with init_lock:
            if app.state.card_store is None:
                app.state.card_store = ReviewStore(root/"local/card_v3_reviews.sqlite3")
            return app.state.card_store

    @app.get("/card-v3", include_in_schema=False)
    def page():
        return FileResponse(Path(__file__).parent/"web/card_v3.html")

    @app.get("/api/card-v3/status")
    def status():
        return {"artifact_present": (root/"artifacts/card_v3_latest.json").is_file(),
                "categories": CATEGORIES, "automatic_actions": False,
                "purpose": "Separate experimental card workflow; human review required",
                "synthetic_labels_human_reviewed": False}

    @app.post("/api/card-v3/requests", status_code=201)
    def submit(payload: NewRequest):
        if not app.state.inference_lock.acquire(blocking=False):
            raise HTTPException(503, "The local model is busy. Retry in a moment.")
        try:
            if app.state.card_router is None:
                if card_factory is not None:
                    app.state.card_router = card_factory(root)
                else:
                    original = app.state.router
                    app.state.card_router = CardRouter(root, shared_encoder=(original.tokenizer, original.encoder))
            analysis = app.state.card_router.route(payload.text)
            if analysis.get("action_executed") is not False or analysis.get("route_recommendation") != "review_required":
                raise ValueError("Review-only model contract violated.")
            return store().create(payload.text, analysis)
        except (FileNotFoundError, ValueError, OSError, KeyError):
            raise HTTPException(503, "The hybrid card model is unavailable. Preserve artifacts, then run train-card-v3 or use the original review desk.") from None
        finally:
            app.state.inference_lock.release()

    @app.get("/api/card-v3/requests")
    def pending(limit: int = Query(20, ge=1, le=50)):
        return store().list("pending", limit, 0)

    @app.get("/api/card-v3/requests/{identity}")
    def detail(identity: str):
        try:
            return store().get(identity)
        except KeyError:
            raise HTTPException(404, "Hybrid card request not found.") from None

    @app.post("/api/card-v3/requests/{identity}/review")
    def review(identity: str, payload: Decision):
        if payload.label not in CATEGORIES:
            raise HTTPException(422, "Choose one of the eight card workflow categories.")
        try:
            item = store().review(identity, payload.label, payload.note, payload.expected_version)
            return {**item, "handoff": handoff(payload.label)}
        except KeyError:
            raise HTTPException(404, "Hybrid card request not found.") from None
        except ReviewConflict as error:
            raise HTTPException(409, str(error)) from None

    return app
