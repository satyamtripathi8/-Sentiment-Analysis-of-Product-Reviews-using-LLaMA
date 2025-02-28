from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import nltk
from nltk.tokenize import word_tokenize
import pickle

nltk.download('punkt')

app = FastAPI()                                                                                                             

# Load saved vocabulary
with open("word_to_index.pkl", "rb") as f:
    word_to_index = pickle.load(f)

# Load the trained LSTM Model
class LSTMSentiment(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=256, output_dim=3):
        super(LSTMSentiment, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        out = self.fc(lstm_out[:, -1, :])  # Take last LSTM output
        return self.softmax(out)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMSentiment(len(word_to_index)).to(device)
model.load_state_dict(torch.load("lstm_sentiment_model.pth", map_location=device))
model.eval()

# Encode review (same as before)
def encode_review(tokens, max_len=50):
    encoded = [word_to_index.get(word, 1) for word in tokens]  # 1 for <UNK>
    return encoded[:max_len] + [0] * (max_len - len(encoded))  # Padding

# Prediction Function
def predict_sentiment(review):
    tokens = word_tokenize(review.lower())
    encoded = encode_review(tokens)
    input_tensor = torch.tensor([encoded], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        sentiment = torch.argmax(output).item()

    return sentiment  # Returns 0, 1, or 2

# Define request body schema
class ReviewRequest(BaseModel):
    review: str

# FastAPI Route
@app.post("/predict")
def predict(review_request: ReviewRequest):
    review = review_request.review

    if not review:
        raise HTTPException(status_code=400, detail="No review text provided")

    sentiment = predict_sentiment(review)
    return {"review": review, "sentiment": sentiment}

# Run FastAPI App
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)