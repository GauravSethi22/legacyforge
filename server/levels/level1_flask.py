from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/items/<int:item_id>', methods=['GET'])
def read_item(item_id):
    if item_id > 0:
        return {"item_id": item_id, "name": "Test Item"}
    return {"error": "Item not found"}, 404

if __name__ == "__main__":
    app.run()
