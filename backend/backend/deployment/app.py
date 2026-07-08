from fastapi import FastAPI
from pydantic import BaseModel
import traceback

from deployment.score import init, run

app = FastAPI(
    title="Support Ticket Classification API",
    description="Support Ticket Prediction API",
    version="1.0"
)


class Ticket(BaseModel):
    text: str


@app.on_event("startup")
def startup():

    init()


@app.get("/")
def home():

    return {
        "status": "running",
        "message": "Support Ticket API"
    }


@app.post("/predict")
def predict(ticket: Ticket):

    try:

        result = run(
            {
                "text": ticket.text
            }
        )

        return result

    except Exception:

        return {
            "status": "error",
            "traceback": traceback.format_exc()
        }