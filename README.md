\# 🌱 AgriSense AI



AgriSense AI is an AI-powered agriculture decision-support system designed to help farmers make data-driven decisions using soil and environmental parameters.



The system combines crop recommendation, adaptive yield prediction, farm production estimation, and PostgreSQL-based prediction history into a single application.



\---



\## 🚀 Features



\### 🌾 Crop Recommendation



AgriSense AI uses a proposed \*\*NSAF (Nutrient-Soil Adaptive Fuzzy)\*\* approach for crop recommendation.



The system analyzes:



\- Nitrogen (N)

\- Phosphorus (P)

\- Potassium (K)

\- Temperature

\- Humidity

\- Soil pH

\- Rainfall



The system returns a ranked list of suitable crops along with their corresponding distances/scores.



Example:



```text

\#1 Jute       → 0.8511

\#2 Rice       → 0.9016

\#3 Cotton     → 1.5554

\#4 Maize      → 1.8249

\#5 Mungbean   → 2.1473

```



\### 📈 Adaptive Yield Prediction



The system predicts agricultural yield in \*\*tonnes per hectare (t/ha)\*\* using soil and environmental parameters.



It also calculates estimated total production according to farm area.



Example:



```text

Predicted Yield      : 16.918 t/ha

Farm Area            : 4 ha

Estimated Production : 67.673 tonnes

```



\### 🗄️ PostgreSQL Database



The application stores:



\- Farm records

\- Predictions

\- Crop recommendations

\- Prediction history



\---



\# 🏗️ System Architecture



```text

&#x20;                   ┌─────────────────────┐

&#x20;                   │   React Frontend    │

&#x20;                   │      Dashboard      │

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;                              │ REST API

&#x20;                              ▼

&#x20;                   ┌─────────────────────┐

&#x20;                   │    FastAPI Backend  │

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;                ┌─────────────┴─────────────┐

&#x20;                │                           │

&#x20;                ▼                           ▼

&#x20;       ┌─────────────────┐        ┌─────────────────┐

&#x20;       │ NSAF Crop Model │        │ Adaptive Yield  │

&#x20;       │ Recommendation  │        │ Prediction Model│

&#x20;       └─────────────────┘        └─────────────────┘

&#x20;                │                           │

&#x20;                └─────────────┬─────────────┘

&#x20;                              ▼

&#x20;                   ┌─────────────────────┐

&#x20;                   │     PostgreSQL      │

&#x20;                   │      Database       │

&#x20;                   └─────────────────────┘

```



\---



\# 🛠️ Technology Stack



\### Frontend

\- React

\- Vite

\- JavaScript

\- HTML

\- CSS

\- Victory

\- Fetch API



\### Backend

\- Python

\- FastAPI

\- Uvicorn

\- SQLAlchemy

\- Pydantic



\### Machine Learning

\- NumPy

\- Pandas

\- Scikit-learn

\- Custom NSAF approach

\- Adaptive Yield Prediction



\### Database

\- PostgreSQL

\- pgAdmin



\### Development Tools

\- Git

\- GitHub

\- Visual Studio Code



\---



\# 📦 Dependencies



The project contains dependency files so that the required packages can be installed on another system.



\### Python Dependencies



Python dependencies are stored in:



```text

requirements.txt

```



Install all Python dependencies with:



```bash

pip install -r requirements.txt

```



This includes the Python libraries required by the backend, database connectivity, data processing, and machine-learning components.



\### Frontend Dependencies



Frontend dependencies are stored in:



```text

frontend/package.json

frontend/package-lock.json

```



Install them with:



```bash

cd frontend

npm install

```



`npm install` automatically downloads the required frontend packages.



\---



\# 💻 Installation



\## 1. Clone the Repository



```bash

git clone https://github.com/Manasvi312003/AgriSense-AI.git

cd AgriSense-AI

```



\---



\## 2. Create Python Virtual Environment



\### Windows



```powershell

python -m venv .venv

.venv\\Scripts\\activate

```



\### Linux / macOS



```bash

python3 -m venv .venv

source .venv/bin/activate

```



\---



\## 3. Install Python Dependencies



From the project root:



```bash

pip install -r requirements.txt

```



\---



\## 4. Install Frontend Dependencies



```bash

cd frontend

npm install

cd ..

```



\---



\# 🗄️ Database Setup



AgriSense AI uses PostgreSQL as its database.



Create the database:



```sql

CREATE DATABASE agrisense\_db;

```



Configure the PostgreSQL connection using environment variables.



> Never commit database passwords, API keys, or other secrets to GitHub.



\---



\# ▶️ Running the Application



\## Start Backend



From the project root:



```bash

python -m uvicorn backend.main:app --reload

```



Backend:



```text

http://127.0.0.1:8000

```



Swagger API documentation:



```text

http://127.0.0.1:8000/docs

```



\---



\## Start Frontend



Open another terminal:



```bash

cd frontend

npm run dev

```



Frontend:



```text

http://localhost:5173

```



\---



\# 🔌 API Endpoints



\### Health Check



```http

GET /

```



\### Crop Recommendation



```http

POST /recommend

```



\### Yield Prediction



```http

POST /predict-yield

```



\### Complete Farm Analysis



```http

POST /analyze

```



The `/analyze` endpoint performs crop recommendation and yield prediction and stores the resulting farm record and predictions in PostgreSQL.



\---



\# 📊 Model Evaluation



\## Crop Recommendation



| Model | Accuracy | Precision | Recall | F1-Score |

|---|---:|---:|---:|---:|

| Random Forest | 0.9932 | 0.9935 | 0.9932 | 0.9932 |

| SVM | 0.9886 | 0.9896 | 0.9886 | 0.9887 |

| KNN | 0.9795 | 0.9804 | 0.9795 | 0.9793 |

| NSAF (Proposed) | 0.9159 | 0.9179 | 0.9159 | 0.9138 |



\## Yield Prediction



| Model | MAE | RMSE | R² |

|---|---:|---:|---:|

| Random Forest Regressor | 1.3939 | 1.7699 | 0.8022 |

| Gradient Boosting | 1.3071 | 1.6600 | 0.8260 |

| Linear Regression | 1.3826 | 1.7509 | 0.8064 |

| Adaptive Yield | 1.3825 | 1.7507 | 0.8065 |



The proposed models are evaluated against commonly used machine-learning baselines to study their predictive behavior.



\---



\# 📁 Project Structure



```text

AgriSense-AI/

│

├── backend/

│   ├── main.py

│   ├── nsaf\_service.py

│   └── yield\_service.py

│

├── database/

│   ├── database.py

│   └── models.py

│

├── frontend/

│   ├── src/

│   ├── package.json

│   └── package-lock.json

│

├── nsaf/

│   ├── nsaf\_core.py

│   ├── recommend\_crop.py

│   ├── predict\_crop.py

│   ├── train\_nsaf.py

│   └── validate\_recommendation.py

│

├── yield/

│   ├── yield\_core.py

│   ├── predict\_yield.py

│   ├── train\_yield.py

│   └── validate\_yield.py

│

├── comparison/

│   └── ...

│

├── data/

│   └── ...

│

├── requirements.txt

├── generate\_yield\_data.py

├── .gitignore

└── README.md

```



\---



\# 🔮 Future Improvements



\- Cloud deployment

\- Real-time weather integration

\- Larger agricultural datasets

\- Farmer-specific recommendations

\- Explainable AI

\- Mobile application

\- Multilingual farmer interface

\- Advanced yield forecasting

\- Weather-based adaptive recommendations



\---



\# 👨‍💻 Author



\*\*Manasvi Kumar\*\*



B.Tech Computer Science \& Engineering



\---



\# 📄 License



This project is developed for educational and research purposes.

