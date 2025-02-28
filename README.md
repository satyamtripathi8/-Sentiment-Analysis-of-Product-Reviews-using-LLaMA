# 🌟 Sentiment Analysis API with FastAPI & LSTM

This is a **Sentiment Analysis API** built using **FastAPI** and an **LSTM-based model trained on product reviews**. It predicts sentiment as:
- **0 → Negative**
- **1 → Neutral**
- **2 → Positive**

The API can be deployed on **Render and Vercel**.

---

## 🚀 Features
✅ Sentiment prediction for product reviews  
✅ Built with **FastAPI + PyTorch (LSTM)**  
✅ Supports **Vercel & Render**  
✅ **Pretrained Model** included for quick inference  
✅ **CORS enabled** for frontend integration  

---

## 💂️🗂 Project Structure

```
📺 Sentiment-Analysis-FastAPI
│-- 📄 app.py                 # FastAPI application
│-- 📄 lstm_model.pth         # Trained LSTM model
│-- 📄 word_to_index.pkl      # Word index for text encoding
│-- 📄 requirements.txt       # Python dependencies
│-- 📄 README.md              # Project documentation
│-- 📄 check_versions.py      # Script to check package versions
│-- 📄 Sentiment analysis for product review.py      # Code for Analyser

```

---

## 🛠 Installation & Setup

### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/yourusername/Sentiment-Analysis-FastAPI.git
cd Sentiment-Analysis-FastAPI
```

### 2️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```



---


#### **Example Response**
```json
{
  "review": "Amazing product, I love it!",
  "sentiment": 2
}
```

---

## 📺 Deployment on Render

### **1️⃣ Deploy via Render Web Service**
1. Go to **[Render.com](https://render.com/)**
2. Click **"New Web Service"**
3. Connect your **GitHub repo**
4. Set **Start Command**:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 10000
   ```
5. Click **Deploy** and get your API URL!

### **2️⃣ Test the Deployed API**
```bash
curl -X POST "https://your-render-app.onrender.com/predict" -H "Content-Type: application/json" -d '{"review": "Not worth buying!"}'
```

---

## 📝 License
This project is open-source and available under the **MIT License**.

---

## 💡 Future Improvements
✅ Improve **model accuracy** using **transformers (BERT/LLaMA)**  
✅ Add **Docker support** for easy deployment  
✅ Build a **frontend (React/Streamlit)** for user interaction  

---

## ⭐ Contributing
Want to improve this project? Feel free to **fork & submit a PR**!

---

### 📚 **Connect with Me**
[GitHub](https://github.com/satyamtripathi08) | [LinkedIn](https://linkedin.com/in/yourprofile) | [Twitter](https://twitter.com/yourhandle)

