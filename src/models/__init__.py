from .model import GPT2forVSB, GPT4TS
from .all4one import ALL4ONE, ALL4ONEonlyTS2VEC

model_factory = {
    "GPT2": GPT2forVSB,
    "GPT4TS": GPT4TS,
    "ALL4ONE": ALL4ONE,
    "ALL4ONEonlyTS2VEC": ALL4ONEonlyTS2VEC,
}
