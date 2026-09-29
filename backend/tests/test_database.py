from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text

from database.connection import get_db


def test_get_db():
    generator = get_db()
    db = next(generator)

    try:
        assert db is not None
    finally:
        try:
            next(generator)
        except StopIteration:
            pass


def test_fastapi_depends_get_db():
    test_app = FastAPI()

    @test_app.get("/test-db")
    def test_db(db=Depends(get_db)):
        db_name = db.execute(text("SELECT DB_NAME()")).scalar()
        return {"database": db_name}

    client = TestClient(test_app)

    response = client.get("/test-db")

    assert response.status_code == 200
    assert response.json()["database"] == "HMNC_PRO"