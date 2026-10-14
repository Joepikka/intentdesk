"""Research-lab endpoints attached to the existing local-only application."""

from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse,JSONResponse,RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from intentdesk.app import NewRequest
from intentdesk.common import read_json
from intentdesk.feedback_lab import directory_for
from intentdesk.feedback_release import improvement_report


def create_lab_app(root: Path):
    """Standalone lab: no MPNet, old intent artifacts or human-pilot database needed."""
    app=FastAPI(title="IntentDesk feedback research lab",docs_url=None,redoc_url=None)
    app.state.inference_lock=Lock()
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=["127.0.0.1","localhost","[::1]"])

    @app.middleware("http")
    async def local_boundary(request: Request,call_next):
        if request.method in {"POST","PUT","PATCH","DELETE"}:
            origin=request.headers.get("origin")
            if request.headers.get("x-intentdesk-client")!="local-web" or (origin and origin!=str(request.base_url).rstrip("/")):
                return JSONResponse({"detail":"Use the local research app."},status_code=403)
            body=bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body)>32768:
                    return JSONResponse({"detail":"Request is too large."},status_code=413)
            request._body=bytes(body)
        result=await call_next(request)
        result.headers.update({"Cache-Control":"no-store","X-Content-Type-Options":"nosniff","Referrer-Policy":"no-referrer",
            "Content-Security-Policy":"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"})
        return result

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request,error):
        return JSONResponse({"detail":"Check the request fields and their length/type."},status_code=422)

    @app.get("/",include_in_schema=False)
    def index():
        return RedirectResponse("/ml-lab")

    app.mount("/assets",StaticFiles(directory=Path(__file__).parent/"web"),name="assets")
    install(app,root)
    return app


def install(app, root: Path, model_factory=None):
    app.state.feedback_models = None

    @app.get("/ml-lab",include_in_schema=False)
    def page():
        return FileResponse(Path(__file__).parent/"web/feedback_lab.html")

    @app.get("/api/ml-lab/status")
    def status():
        ready, report, selected = False, None, None
        if (root/"artifacts/feedback_lab_latest.json").exists():
            directory = directory_for(root)
            if (directory/"metadata.json").exists():
                metadata = read_json(directory/"metadata.json")
                ready, selected = True, metadata["selected_classifier"]
                marker = directory/"final_experiment.json"
                if marker.exists() and read_json(marker).get("status")=="completed":
                    relative = read_json(marker)["report"]
                    path = (root/relative).resolve()
                    if not path.is_relative_to((root/"reports/feedback_lab_runs").resolve()):
                        raise HTTPException(503,"Research report path is invalid.")
                    result = read_json(path)
                    report = {"classification":result["classification"],"regression":result["regression"],
                        "unsupervised":result["unsupervised"],"data":metadata["data"],
                        "scratch_models":{n:metadata["development_classification"][n]["training"] for n in ("scratch_small","scratch_large")}}
        try:
            improved=improvement_report(root)
        except (OSError,ValueError,KeyError):
            raise HTTPException(503,"The completed improvement release is incompatible. Check its local report.") from None
        return {"ready":ready,"selected_classifier":selected,"final_report":report,
                "active_classifier":improved["selected_classifier"] if improved else selected,
                "improvement_report":None if improved is None else {
                    "classification":improved["classification"],"selected_classifier":improved["selected_classifier"],
                    "fresh_check":improved["training"]["protocol"]["fresh_check"]},
                "purpose":"Public product-review research; separate from card-intent routing",
                "automatic_actions":False,"business_value_established":False}

    @app.post("/api/ml-lab/analyze")
    def analyze(payload: NewRequest):
        if not app.state.inference_lock.acquire(blocking=False):
            raise HTTPException(503,"The local model is busy. Retry in a moment.")
        try:
            if app.state.feedback_models is None:
                if model_factory is None:
                    from intentdesk.feedback_release import FeedbackRelease
                    factory = FeedbackRelease
                else:
                    factory = model_factory
                app.state.feedback_models = factory(root)
            result = app.state.feedback_models.analyze(payload.text)
            if result.get("automatic_actions") is not False:
                raise ValueError("Research-only action contract violated.")
            return result
        except (OSError,ValueError,KeyError):
            raise HTTPException(503,"Feedback models are unavailable or incompatible. Check the local training report.") from None
        finally:
            app.state.inference_lock.release()
