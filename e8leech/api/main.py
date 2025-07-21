from fastapi import FastAPI
from pydantic import BaseModel
from e8leech.core.golay_code import E8Lattice, LeechLattice
import numpy as np

app = FastAPI()

class Vector(BaseModel):
    vector: list[float]

@app.get("/e8/kissing_number")
def get_e8_kissing_number():
    e8 = E8Lattice()
    return {"kissing_number": e8.kissing_number()}

@app.post("/e8/closest_vector")
def get_e8_closest_vector(vector: Vector):
    e8 = E8Lattice()
    point = np.array(vector.vector)
    closest = e8.closest_vector(point)
    return {"closest_vector": closest.tolist()}

@app.get("/leech/kissing_number")
def get_leech_kissing_number():
    leech = LeechLattice()
    return {"kissing_number": leech.kissing_number()}

@app.post("/leech/closest_vector")
def get_leech_closest_vector(vector: Vector):
    leech = LeechLattice()
    point = np.array(vector.vector)
    closest = leech.closest_vector(point)
    return {"closest_vector": closest.tolist()}
