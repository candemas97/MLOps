from pydantic import BaseModel

class PENGUIN(BaseModel):
    culmenLen: list[float]
    culmenDepth: list[float]
    flipperLen: list[float]
    bodyMass: list[float]
    sex: list[str]
    delta15N: list[float]
    delta13C: list[float]