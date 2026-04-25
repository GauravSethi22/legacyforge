from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class ItemResponse(BaseModel):
    item_id: int
    name: str

@app.get("/items/{item_id}", response_model=ItemResponse)
async def read_item(item_id: int):
    if item_id > 0:
        return ItemResponse(item_id=item_id, name="Test Item")
    raise HTTPException(status_code=404, detail="Item not found")
