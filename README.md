# MHTCET-College-Predictor
https://mhtcet-college-predictor.streamlit.app/

# 🎓 MHTCET Admission & College Predictor — EduPredict.AI

An interactive, Machine Learning-powered web application designed to predict engineering college branch allocations for the Maharashtra Common Entrance Test (MHTCET). Built with **Python**, **XGBoost**, and **Streamlit**, this tool analyzes candidate percentiles, quota categories, reservation types, and target institutes to infer admission probabilities.

---

## ✨ Features

* **AI Classification Engine**: Leverages trained XGBoost classification models to predict allocated engineering courses based on historical admission data.
* **10 Dynamic Glassmorphism Themes**: Includes 10 customizable themes (**Light Mode**, **Dark Cyberpunk**, **Luxe Gold**, **Emerald Luxe**, **Dracula Night**, **Neon Vaporwave**, **Nordic Frost**, **Sunset Amber**, **Deep Ocean**, and **Monochrome Obsidian**).
* **Confidence Breakdown**: Displays top-5 predicted branch matches alongside animated percentage probability gauges.
* **Safe Encoding Guards**: Built-in fallback handlers (`safe_transform` and `safe_inverse_transform`) to handle out-of-bounds predicted indices and unseen labels smoothly.
* **Session History Tracker**: Keeps a running log of past inferences conducted during the active session.
* **Zero Web-Server Overhead**: Optimized natively for Streamlit Cloud deployment with `@st.cache_resource` for high performance.

---

## 📁 Repository Structure

```text
├── app.py                          # Main Streamlit web application & inference logic
├── requirements.txt                # Pinned Python dependency versions
├── README.md                       # Project documentation
├── institute_model_compressed.pkl.gz  # Compressed XGBoost model file
├── course_model.pkl                # Primary course classification model
├── gender_encoder.pkl              # LabelEncoder for Gender
├── category_encoder.pkl            # LabelEncoder for Reservation Category
├── seat_encoder.pkl                # LabelEncoder for Seat Allocation Quota
├── course_encoder.pkl              # LabelEncoder for Engineering Courses
└── institute_encoder.pkl           # LabelEncoder for Target Institutes


🛠️ How It Works1. Data Ingestion & Fallback LayerWhen the app launches, @st.cache_resource safely loads the pickled/gzipped ML models and LabelEncoder objects into memory. If any encoder file is missing or contains incomplete classes, the app seamlessly falls back to pre-populated native arrays covering over 180 institutes, 90 courses, and 80 categories.2. Candidate InputsThe user specifies 5 core parameters in the UI dashboard:Gender Category (e.g., Male, Female)Reservation Category (e.g., OPEN, OBC, SC, ST, SEBC)MHTCET Percentile Score (e.g., 98.4521)Seat Allocation Quota (e.g., GOPENS, GSCS, TFWS, EWS)Target Institute Name (e.g., COEP Technological University, VJTI Mumbai, PICT Pune)3. Encoding & Inference PipelineInput string parameters are passed into safe_transform() to map categorical strings to their trained integer representations without throwing ValueError exceptions for unseen labels.The parameters are constructed into a feature vector:$$\text{Features} = [\text{Gender}, \text{Category}, \text{Percentile}, \text{Seat Quota}, \text{Institute}]$$The feature array is passed to the XGBoost model via .predict() and .predict_proba().The predicted integer index is mapped back to the branch title using safe_inverse_transform(), preventing IndexError crashes if the index exceeds encoder bounds.4. Output RenderThe interface updates dynamically to show:The Most Likely Allocated Engineering Branch.A breakdown of the Top 5 Branch Probabilities rendered as visual progress bars.A record entry added to the Session Prediction History table.🚀 Local Installation & SetupTo run this project locally on your machine:Clone the repository:Bashgit clone [https://github.com/](https://github.com/)<your-username>/mhtcet-college-predictor.git
cd mhtcet-college-predictor
Create and activate a virtual environment:Bashpython -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
Install dependencies:Bashpip install -r requirements.txt
Launch the Streamlit app:Bashstreamlit run app.py
⚙️ DependenciesThis project relies on the following pinned library versions for maximum stability:Plaintextstreamlit==1.31.0
numpy==1.26.4
pandas==2.2.0
scikit-learn==1.6.1
xgboost==2.0.3
🌐 DeploymentThis application is configured for deployment on Streamlit Cloud:Push all files to your GitHub repository.Connect your repository to share.streamlit.io.Set the Main File Path to app.py.Deploy!
