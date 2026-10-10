"""Single-process, loopback-only review application. No paid services."""

from contextlib import asynccontextmanager
from pathlib import Path
import sqlite3
from threading import Lock
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from intentdesk.predict import validate_text
from intentdesk.common import read_json
from intentdesk.reviews import ReviewConflict, ReviewStore
from intentdesk.pilot import CATEGORIES, PROTOCOL, PilotStore


class NewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: StrictStr = Field(min_length=1, max_length=4000)

    @field_validator("text")
    @classmethod
    def clean(cls, value):
        return validate_text(value)


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: StrictStr = Field(min_length=1, max_length=100)
    note: StrictStr = Field(default="", max_length=1000)
    expected_version: int = Field(ge=0, strict=True)


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: StrictStr = Field(min_length=1, max_length=1000)


class PilotSession(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reviewer: StrictStr = Field(min_length=3, max_length=40, pattern=r"^[A-Za-z0-9_-]+$")
    kind: Literal["human", "smoke"] = "human"


class PilotDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    trial_id: str = Field(min_length=1, max_length=100)
    label: str = Field(min_length=1, max_length=100)
    interrupted: bool = False


def create_app(root: Path, router_factory=None, database: Path | None = None) -> FastAPI:
    if router_factory is None:
        from intentdesk.scope import ScopeRouter
        router_factory = ScopeRouter

    @asynccontextmanager
    async def lifespan(app):
        app.state.router = router_factory(root)
        app.state.store = ReviewStore(database or root/"local/reviews.sqlite3")
        app.state.pilot = PilotStore(root/"local/pilot.sqlite3", root, app.state.router)
        app.state.inference_lock = Lock()
        app.state.search_lock = Lock()
        app.state.searcher = None
        if (root/"data/retrieval/msmarco-v1/search_manifest.json").is_file():
            from intentdesk.search import Searcher
            app.state.searcher = Searcher(root)
        yield

    app = FastAPI(title="IntentDesk local review API", version="0.2.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])

    @app.middleware("http")
    async def local_boundary(request: Request, call_next):
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            origin = request.headers.get("origin")
            expected_origin = str(request.base_url).rstrip("/")
            if request.headers.get("x-intentdesk-client") != "local-web" or (origin and origin != expected_origin):
                return JSONResponse({"detail": "Use the local IntentDesk app to make changes."}, status_code=403)
            # Bound the actual streamed body as well as Content-Length; no input is logged.
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 32768:
                    return JSONResponse({"detail": "Request body is too large."}, status_code=413)
            request._body = bytes(body)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        return JSONResponse({"detail": "Check the request fields and their length/type."}, status_code=422)

    @app.exception_handler(sqlite3.Error)
    async def database_error(request, error):
        return JSONResponse({"detail": "The local review database is unavailable. Retry after closing other writers."}, status_code=503)

    @app.get("/api/status")
    def status():
        router = app.state.router
        return {"mode": "local_review", "policy_version": router.metadata["run_id"],
                "development_gate_passed": router.metadata["gate"]["passed"],
                "automatic_actions": False,
                "test_evaluated": (root/"artifacts/final_experiment.json").is_file() and
                                  read_json(root/"artifacts/final_experiment.json").get("test_exposed", False),
                "counts": app.state.store.counts(), "classes": list(router.models.classes),
                "review_labels": ["oos", "uncertain"],
                "scope_assessment_ready": getattr(router, "scope", None) is not None,
                "feedback_used_for_training": False}

    @app.get("/api/search/status")
    def search_status():
        searcher = app.state.searcher
        return {"ready": searcher is not None, "passages": searcher.manifest["passages"] if searcher else 0,
                "purpose": "Separate non-commercial research benchmark; not customer support documentation",
                "answer_generated": False}

    @app.post("/api/search")
    def search(payload: SearchRequest):
        if app.state.searcher is None:
            raise HTTPException(503, "The local search index is not ready. Build it, then restart the app.")
        if not app.state.search_lock.acquire(blocking=False):
            raise HTTPException(503, "Local search is busy. Retry in a moment.")
        try:
            return app.state.searcher.search(payload.query)
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        except TimeoutError as error:
            raise HTTPException(504, str(error)) from None
        finally:
            app.state.search_lock.release()

    @app.post("/api/requests", status_code=201)
    def submit(payload: NewRequest):
        if not app.state.inference_lock.acquire(blocking=False):
            raise HTTPException(503, "The local model is busy. Retry in a moment.")
        try:
            analysis = app.state.router.route(payload.text)
        finally:
            app.state.inference_lock.release()
        # The app is a review queue even if a future offline experiment passes a gate.
        analysis["route_recommendation"] = "review_required"
        analysis["action_executed"] = False
        return app.state.store.create(payload.text, analysis)

    @app.get("/api/requests")
    def queue(status: Literal["pending", "reviewed"] = "pending", limit: int = Query(20, ge=1, le=50),
              offset: int = Query(0, ge=0)):
        return app.state.store.list(status, limit, offset)

    @app.get("/api/requests/{request_id}")
    def detail(request_id: str):
        try:
            return app.state.store.get(request_id)
        except KeyError:
            raise HTTPException(404, "Request not found.") from None

    @app.post("/api/requests/{request_id}/review")
    def review(request_id: str, payload: Decision):
        if payload.label not in {*app.state.router.models.classes, "oos", "uncertain"}:
            raise HTTPException(422, "Choose a supported intent, oos, or uncertain.")
        try:
            return app.state.store.review(request_id, payload.label, payload.note, payload.expected_version)
        except KeyError:
            raise HTTPException(404, "Request not found.") from None
        except ReviewConflict as error:
            raise HTTPException(409, str(error)) from None

    @app.get("/api/pilot")
    def pilot_config():
        return {"categories": CATEGORIES, "protocol": PROTOCOL, "run_id": app.state.pilot.run_id,
                "demonstration_data": True, "automatic_actions": False}

    def pilot_result(operation, *args):
        try:
            return operation(*args)
        except KeyError:
            raise HTTPException(404, "Pilot session not found.") from None
        except ReviewConflict as error:
            raise HTTPException(409, str(error)) from None
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    @app.post("/api/pilot/sessions", status_code=201)
    def pilot_start(payload: PilotSession):
        return pilot_result(app.state.pilot.start, payload.reviewer, payload.kind)

    @app.get("/api/pilot/sessions/{session_id}")
    def pilot_session(session_id: str):
        return pilot_result(app.state.pilot.view, session_id)

    @app.post("/api/pilot/sessions/{session_id}/next")
    def pilot_next(session_id: str):
        if not app.state.inference_lock.acquire(blocking=False):
            raise HTTPException(503, "The local model is busy. Retry in a moment.")
        try:
            return pilot_result(app.state.pilot.next, session_id)
        finally:
            app.state.inference_lock.release()

    @app.post("/api/pilot/sessions/{session_id}/decision")
    def pilot_decide(session_id: str, payload: PilotDecision):
        return pilot_result(app.state.pilot.decide, session_id, payload.trial_id, payload.label, payload.interrupted)

    @app.get("/api/pilot/report")
    def pilot_report():
        return app.state.pilot.report()

    web = Path(__file__).parent/"web"
    app.mount("/assets", StaticFiles(directory=web), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(web/"index.html")

    @app.get("/pilot", include_in_schema=False)
    def pilot_page():
        return FileResponse(web/"pilot.html")

    return app
