from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel

app = FastAPI()

class ItemResponse(BaseModel):
    item_id: int
    name: str

@app.get("/items/{item_id}", response_model=ItemResponse)
async def read_item(item_id: int = Path(..., description="The ID of the item to retrieve")):
    if item_id <= 0:
        raise HTTPException(status_code=422, detail="item_id must be positive")
    if item_id > 1000:
        raise HTTPException(status_code=404, detail="Item not found")
    return ItemResponse(item_id=item_id, name=f"Item {item_id}")
