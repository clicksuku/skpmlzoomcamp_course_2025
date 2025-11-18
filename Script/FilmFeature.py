from pydantic import BaseModel

class FilmFeature(BaseModel):
    runtime: int
    popularity: float
    vote_avg: float
    log_budget: float
