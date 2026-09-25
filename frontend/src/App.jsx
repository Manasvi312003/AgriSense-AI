import { useEffect, useState } from "react";
import "./App.css";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

function App() {
  // =========================================================
  // FARM INPUT DATA
  // =========================================================

  const [formData, setFormData] = useState({
    N: "",
    P: "",
    K: "",
    temperature: "",
    humidity: "",
    ph: "",
    rainfall: "",
    area: "",
  });

  // =========================================================
  // ANALYSIS RESULT
  // =========================================================

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // =========================================================
  // DATABASE HISTORY
  // =========================================================

  const [history, setHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  // =========================================================
  // HANDLE INPUT CHANGE
  // =========================================================

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  // =========================================================
  // LOAD DATABASE HISTORY
  // =========================================================

  const loadHistory = async () => {
    try {
      const response = await fetch(
        "http://127.0.0.1:8000/history"
      );

      if (!response.ok) {
        throw new Error("Failed to load history");
      }

      const data = await response.json();

      setHistory(data.history || []);

    } catch (error) {
      console.error("History error:", error);
    }
  };

  // =========================================================
  // LOAD HISTORY WHEN PAGE OPENS
  // =========================================================

  useEffect(() => {
    loadHistory();
  }, []);
  useEffect(() => {
    fetchHistory();
  }, []);

  // =========================================================
  // FARM ANALYSIS
  // =========================================================

  const handleSubmit = async (e) => {
    e.preventDefault();

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/analyze",
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",
          },

          body: JSON.stringify({
            N: Number(formData.N),
            P: Number(formData.P),
            K: Number(formData.K),
            temperature: Number(formData.temperature),
            humidity: Number(formData.humidity),
            ph: Number(formData.ph),
            rainfall: Number(formData.rainfall),
            area: Number(formData.area),
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Prediction failed"
        );
      }

      // Show current analysis result
      setResult(data);

      // IMPORTANT:
      // Refresh database history AFTER analysis is saved
      await loadHistory();

    } catch (error) {
      console.error(
        "AgriSense AI Error:",
        error
      );

      alert(
        "Unable to connect to AgriSense AI backend."
      );

    } finally {
      setLoading(false);
    }
  };

  const fetchHistory = async () => {
  setHistoryLoading(true);

  try {
    const response = await fetch(
      "http://127.0.0.1:8000/history?limit=20"
    );

    if (!response.ok) {
      throw new Error("Failed to fetch history");
    }

    const data = await response.json();

    setHistory(data.history || []);

  } catch (error) {
    console.error("History error:", error);
  }

  setHistoryLoading(false);
};

  // =========================================================
  // UI
  // =========================================================

  return (
    <div className="app">

      {/* =================================================
          NAVBAR
      ================================================= */}

      <header className="navbar">

        <div className="logo">
          🌱 AgriSense AI
        </div>

        <div className="nav-status">
          NSAF • Adaptive Intelligence
        </div>

      </header>


      <main>

        {/* =================================================
            HERO
        ================================================= */}

        <section className="hero">

          <p className="tag">
            AI-POWERED SMART AGRICULTURE
          </p>

          <h1>
            Intelligent Farming,
            <span> Powered by AI</span>
          </h1>

          <p className="hero-text">
            Get intelligent crop recommendations and
            adaptive yield predictions using AgriSense AI.
          </p>

        </section>


        {/* =================================================
            DASHBOARD
        ================================================= */}

        <section className="dashboard">

          {/* =================================================
              FARM INPUT CARD
          ================================================= */}

          <div className="card input-card">

            <h2>
              🌾 Farm Information
            </h2>

            <p className="card-subtitle">
              Enter your field conditions
            </p>


            <form onSubmit={handleSubmit}>

              <div className="grid">

                <Input
                  label="Nitrogen (N)"
                  name="N"
                  value={formData.N}
                  onChange={handleChange}
                />

                <Input
                  label="Phosphorus (P)"
                  name="P"
                  value={formData.P}
                  onChange={handleChange}
                />

                <Input
                  label="Potassium (K)"
                  name="K"
                  value={formData.K}
                  onChange={handleChange}
                />

                <Input
                  label="Temperature (°C)"
                  name="temperature"
                  value={formData.temperature}
                  onChange={handleChange}
                />

                <Input
                  label="Humidity (%)"
                  name="humidity"
                  value={formData.humidity}
                  onChange={handleChange}
                />

                <Input
                  label="pH"
                  name="ph"
                  value={formData.ph}
                  onChange={handleChange}
                />

                <Input
                  label="Rainfall (mm)"
                  name="rainfall"
                  value={formData.rainfall}
                  onChange={handleChange}
                />

                <Input
                  label="Area (hectares)"
                  name="area"
                  value={formData.area}
                  onChange={handleChange}
                />

              </div>


              <button
                type="submit"
                disabled={loading}
              >
                {loading
                  ? "Analyzing..."
                  : "Analyze Farm →"}
              </button>

            </form>

          </div>


          {/* =================================================
              RESULT CARD
          ================================================= */}

          <div className="card result-card">

            <h2>
              🤖 AI Analysis
            </h2>


            {/* =================================================
                EMPTY STATE
            ================================================= */}

            {!result && !loading && (

              <div className="empty">

                <div className="empty-icon">
                  🌱
                </div>

                <h3>
                  Ready to analyze
                </h3>

                <p>
                  Enter your farm data and let
                  AgriSense AI generate insights.
                </p>

              </div>

            )}


            {/* =================================================
                LOADING STATE
            ================================================= */}

            {loading && (

              <div className="empty">

                <div className="loader">
                  ⟳
                </div>

                <h3>
                  AI is analyzing...
                </h3>

                <p>
                  NSAF and Adaptive Yield Learner
                  are processing your farm conditions.
                </p>

              </div>

            )}


            {/* =================================================
                RESULTS
            ================================================= */}

            {result && (

              <div className="results">

                {/* =================================================
                    RECOMMENDED CROP
                ================================================= */}

                <div className="result-main">

                  <p>
                    RECOMMENDED CROP
                  </p>

                  <h1>
                    🌾{" "}
                    {result.crop_recommendation.recommended_crop}
                  </h1>

                </div>


                {/* =================================================
                    YIELD
                ================================================= */}

                <div className="yield-box">

                  <span>
                    Predicted Yield
                  </span>

                  <strong>
                    {Number(
                      result.yield_prediction.predicted_yield
                    ).toFixed(3)}
                  </strong>

                  <small>
                    {result.yield_prediction.yield_unit}
                  </small>

                </div>


                {/* =================================================
                    FARM AREA + TOTAL PRODUCTION
                ================================================= */}

                <div className="production-grid">

                  <div className="production-box">

                    <span>
                      Farm Area
                    </span>

                    <strong>
                      {result.yield_prediction.area}
                    </strong>

                    <small>
                      {result.yield_prediction.area_unit}
                    </small>

                  </div>


                  <div className="production-box">

                    <span>
                      Estimated Production
                    </span>

                    <strong>
                      {Number(
                        result.yield_prediction.estimated_production
                      ).toFixed(3)}
                    </strong>

                    <small>
                      {result.yield_prediction.production_unit}
                    </small>

                  </div>

                </div>


                {/* =================================================
                    TOP PREDICTIONS
                ================================================= */}

                <h3 className="prediction-title">
                  Top Crop Predictions
                </h3>


                <div className="prediction-list">

                  {result.crop_recommendation.top_predictions.map(
                    (item, index) => (

                      <div
                        className="prediction"
                        key={item.crop}
                      >

                        <span>
                          #{index + 1}
                        </span>

                        <strong>
                          {item.crop}
                        </strong>

                        <small>
                          distance: {item.distance}
                        </small>

                      </div>

                    )
                  )}

                </div>


                {/* =================================================
                    DATABASE STATUS
                ================================================= */}

                {result.database_status === "saved" && (

                  <div className="database-status">
                    ✓ Analysis saved successfully
                  </div>

                )}

              </div>

            )}

          </div>

        </section>



{/* =====================================================
    MODEL COMPARISON
===================================================== */}

<section className="comparison-section">

  <div className="comparison-header">
    <p className="tag">MODEL EVALUATION</p>

    <h2>
      AgriSense AI vs Existing Models
    </h2>

    <p>
      Performance comparison of the proposed NSAF and
      Adaptive Yield models with conventional machine learning models.
    </p>
  </div>


  {/* ================= CROP COMPARISON ================= */}

  <div className="comparison-card">

    <div className="comparison-title">
      <div>
        <h3>🌾 Crop Recommendation</h3>
        <p>Classification performance</p>
      </div>
    </div>

    <div className="table-wrapper">

      <table className="comparison-table">

        <thead>
          <tr>
            <th>Model</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1 Score</th>
          </tr>
        </thead>

        <tbody>

          <tr>
            <td>Random Forest</td>
            <td>99.32%</td>
            <td>99.35%</td>
            <td>99.32%</td>
            <td>99.32%</td>
          </tr>

          <tr>
            <td>SVM</td>
            <td>98.86%</td>
            <td>98.96%</td>
            <td>98.86%</td>
            <td>98.87%</td>
          </tr>

          <tr>
            <td>KNN</td>
            <td>97.95%</td>
            <td>98.04%</td>
            <td>97.95%</td>
            <td>97.93%</td>
          </tr>

          <tr className="proposed-row">
            <td>
              <strong>NSAF (Proposed)</strong>
            </td>
            <td>91.59%</td>
            <td>91.79%</td>
            <td>91.59%</td>
            <td>91.38%</td>
          </tr>

        </tbody>

      </table>

    </div>

  </div>


  {/* ================= YIELD COMPARISON ================= */}

  <div className="comparison-card">

    <div className="comparison-title">

      <div>
        <h3>📈 Yield Prediction</h3>
        <p>Regression performance — lower MAE/RMSE is better, higher R² is better</p>
      </div>

    </div>


    <div className="table-wrapper">

      <table className="comparison-table">

        <thead>

          <tr>
            <th>Model</th>
            <th>MAE (t/ha)</th>
            <th>RMSE (t/ha)</th>
            <th>R²</th>
          </tr>

        </thead>


        <tbody>

          <tr>
            <td>Random Forest Regressor</td>
            <td>1.3939</td>
            <td>1.7699</td>
            <td>0.8022</td>
          </tr>

          <tr>
            <td>Gradient Boosting</td>
            <td>1.3071</td>
            <td>1.6600</td>
            <td>0.8260</td>
          </tr>

          <tr>
            <td>Linear Regression</td>
            <td>1.3826</td>
            <td>1.7509</td>
            <td>0.8064</td>
          </tr>

          <tr className="proposed-row">
            <td>
              <strong>Adaptive Yield (Proposed)</strong>
            </td>
            <td>1.3825</td>
            <td>1.7507</td>
            <td>0.8065</td>
          </tr>

        </tbody>

      </table>

    </div>

  </div>
  {/* ================= VISUAL BARS ================= */}

  <div className="comparison-card">

    <h3>📊 Performance Overview</h3>

    <div className="metric-grid">

      <div className="metric-box">

        <h4>Crop Accuracy</h4>

        <div className="metric-row">
          <span>Random Forest</span>
          <div className="bar">
            <div style={{ width: "99.32%" }}></div>
          </div>
          <b>99.32%</b>
        </div>

        <div className="metric-row">
          <span>SVM</span>
          <div className="bar">
            <div style={{ width: "98.86%" }}></div>
          </div>
          <b>98.86%</b>
        </div>

        <div className="metric-row">
          <span>KNN</span>
          <div className="bar">
            <div style={{ width: "97.95%" }}></div>
          </div>
          <b>97.95%</b>
        </div>

        <div className="metric-row proposed-metric">
          <span>NSAF</span>
          <div className="bar">
            <div style={{ width: "91.59%" }}></div>
          </div>
          <b>91.59%</b>
        </div>

      </div>


      <div className="metric-box">

        <h4>Yield R²</h4>

        <div className="metric-row">
          <span>Random Forest</span>
          <div className="bar">
            <div style={{ width: "80.22%" }}></div>
          </div>
          <b>0.8022</b>
        </div>

        <div className="metric-row">
          <span>Gradient Boosting</span>
          <div className="bar">
            <div style={{ width: "82.60%" }}></div>
          </div>
          <b>0.8260</b>
        </div>

        <div className="metric-row">
          <span>Linear Regression</span>
          <div className="bar">
            <div style={{ width: "80.64%" }}></div>
          </div>
          <b>0.8064</b>
        </div>

        <div className="metric-row proposed-metric">
          <span>Adaptive Yield</span>
          <div className="bar">
            <div style={{ width: "80.65%" }}></div>
          </div>
          <b>0.8065</b>
        </div>

      </div>

    </div>

  </div>

</section>       
{ /* =========================================================
DATA VISUALIZATION
========================================================= */}

<section className="visualization-section">

  <div className="section-heading">
    <p className="tag">DATA INSIGHTS</p>

    <h2>
      Farm Analysis Dashboard
    </h2>

    <p>
      Historical predictions generated by AgriSense AI
    </p>
  </div>


  {historyLoading ? (

    <div className="visualization-loading">
      Loading database insights...
    </div>

  ) : history.length === 0 ? (

    <div className="visualization-empty">
      No historical farm data available yet.
    </div>

  ) : (

    <>

      {/* SUMMARY CARDS */}

      <div className="stats-grid">

        <div className="stat-card">
          <span>Total Analyses</span>
          <strong>{history.length}</strong>
        </div>

        <div className="stat-card">
          <span>Latest Crop</span>
          <strong>
            {history[0]?.recommended_crop || "—"}
          </strong>
        </div>

        <div className="stat-card">
          <span>Latest Yield</span>
          <strong>
            {history[0]?.predicted_yield ?? "—"} t/ha
          </strong>
        </div>

        <div className="stat-card">
          <span>Latest Area</span>
          <strong>
            {history[0]?.inputs?.area ?? "—"} ha
          </strong>
        </div>

      </div>


      {/* YIELD TREND */}

      <div className="chart-card">

        <h3>Predicted Yield Trend</h3>

        <p>
          Yield predictions across recent farm analyses
        </p>

        <ResponsiveContainer
          width="100%"
          height={350}
        >

          <LineChart
            data={[...history].reverse().map(
              (item, index) => ({
                analysis: index + 1,
                yield: item.predicted_yield,
              })
            )}
          >

            <CartesianGrid strokeDasharray="3 3" />

            <XAxis
              dataKey="analysis"
              label={{
                value: "Analysis",
                position: "insideBottom",
                offset: -5,
              }}
            />

            <YAxis
              label={{
                value: "Yield (t/ha)",
                angle: -90,
                position: "insideLeft",
              }}
            />

            <Tooltip />

            <Line
              type="monotone"
              dataKey="yield"
              stroke="#278642"
              strokeWidth={3}
              dot={{ r: 4 }}
            />

          </LineChart>

        </ResponsiveContainer>

      </div>


      {/* NPK CHART */}

      <div className="chart-card">

        <h3>Nutrient Analysis</h3>

        <p>
          Nitrogen, Phosphorus and Potassium used in recent analyses
        </p>

        <ResponsiveContainer
          width="100%"
          height={350}
        >

          <BarChart
            data={[...history]
              .slice(0, 10)
              .reverse()
              .map((item, index) => ({
                analysis: index + 1,
                N: item.inputs?.N || 0,
                P: item.inputs?.P || 0,
                K: item.inputs?.K || 0,
              }))}
          >

            <CartesianGrid strokeDasharray="3 3" />

            <XAxis dataKey="analysis" />

            <YAxis />

            <Tooltip />

            <Bar
              dataKey="N"
              fill="#278642"
            />

            <Bar
              dataKey="P"
              fill="#e0a12b"
            />

            <Bar
              dataKey="K"
              fill="#4b83c4"
            />

          </BarChart>

        </ResponsiveContainer>

      </div>


      {/* RAINFALL VS YIELD */}

      <div className="chart-card">

        <h3>Rainfall vs Predicted Yield</h3>

        <p>
          Relationship between rainfall conditions and predicted yield
        </p>

        <ResponsiveContainer
          width="100%"
          height={350}
        >

          <LineChart
            data={[...history]
              .reverse()
              .map((item) => ({
                rainfall: item.inputs?.rainfall || 0,
                yield: item.predicted_yield || 0,
              }))}
          >

            <CartesianGrid strokeDasharray="3 3" />

            <XAxis
              dataKey="rainfall"
              label={{
                value: "Rainfall (mm)",
                position: "insideBottom",
                offset: -5,
              }}
            />

            <YAxis
              label={{
                value: "Yield (t/ha)",
                angle: -90,
                position: "insideLeft",
              }}
            />

            <Tooltip />

            <Line
              type="monotone"
              dataKey="yield"
              stroke="#278642"
              strokeWidth={3}
              dot={{ r: 4 }}
            />

          </LineChart>

        </ResponsiveContainer>

      </div>


      {/* HISTORY TABLE */}

      <div className="history-card">

        <h3>Recent Farm Analyses</h3>

        <div className="table-wrapper">

          <table>

            <thead>

              <tr>
                <th>#</th>
                <th>Crop</th>
                <th>Yield</th>
                <th>Area</th>
                <th>Rainfall</th>
                <th>Date</th>
              </tr>

            </thead>

            <tbody>

              {history.slice(0, 10).map(
                (item, index) => (

                  <tr key={item.farm_record_id}>

                    <td>
                      {index + 1}
                    </td>

                    <td>
                      <strong>
                        {item.recommended_crop}
                      </strong>
                    </td>

                    <td>
                      {item.predicted_yield} t/ha
                    </td>

                    <td>
                      {item.inputs?.area} ha
                    </td>

                    <td>
                      {item.inputs?.rainfall} mm
                    </td>

                    <td>
                      {item.created_at
                        ? new Date(
                            item.created_at
                          ).toLocaleDateString()
                        : "—"}
                    </td>

                  </tr>

                )
              )}

            </tbody>

          </table>

        </div>

      </div>

    </>

  )}

</section>


        {/* =========================================================
            DATABASE / ANALYSIS HISTORY
        ========================================================= */}

        <section className="history-section">

          <div className="history-header">

            <div>
              <h2>
                📊 Analysis History
              </h2>

              <p>
                Previous farm analyses stored in PostgreSQL
              </p>
            </div>


            <button
              className="refresh-btn"
              onClick={loadHistory}
            >
              ↻ Refresh
            </button>

          </div>


          {/* =================================================
              NO RECORDS
          ================================================= */}

          {history.length === 0 ? (

            <div className="empty-history">

              <div className="empty-history-icon">
                🌱
              </div>

              <h3>
                No analysis records found
              </h3>

              <p>
                Run a farm analysis to create your
                first database record.
              </p>

            </div>

          ) : (

            /* =================================================
               HISTORY TABLE
            ================================================= */

            <div className="history-table-wrapper">

              <table className="history-table">

                <thead>

                  <tr>

                    <th>ID</th>

                    <th>N</th>

                    <th>P</th>

                    <th>K</th>

                    <th>Temperature</th>

                    <th>Humidity</th>

                    <th>pH</th>

                    <th>Rainfall</th>

                    <th>Area</th>

                    <th>Recommended Crop</th>

                    <th>Predicted Yield</th>

                  </tr>

                </thead>


                <tbody>

                  {history.map((record) => (

                    <tr
                      key={record.farm_record_id}
                    >

                      <td>
                        #{record.farm_record_id}
                      </td>

                      <td>
                        {record.inputs.N}
                      </td>

                      <td>
                        {record.inputs.P}
                      </td>

                      <td>
                        {record.inputs.K}
                      </td>

                      <td>
                        {record.inputs.temperature}
                      </td>

                      <td>
                        {record.inputs.humidity}
                      </td>

                      <td>
                        {record.inputs.ph}
                      </td>

                      <td>
                        {record.inputs.rainfall}
                      </td>

                      <td>
                        {record.inputs.area} ha
                      </td>

                      <td>

                        <strong className="crop-name">
                          {record.recommended_crop}
                        </strong>

                      </td>

                      <td>

                        <strong>
                          {Number(
                            record.predicted_yield
                          ).toFixed(3)}
                        </strong>

                        <span className="unit-text">
                          {" "}t/ha
                        </span>

                      </td>

                    </tr>

                  ))}

                </tbody>

              </table>

            </div>

          )}

        </section>

      </main>

    </div>
  );
}


// =========================================================
// INPUT COMPONENT
// =========================================================

function Input({
  label,
  name,
  value,
  onChange,
}) {

  return (

    <div className="input-group">

      <label>
        {label}
      </label>

      <input
        type="number"
        step="any"
        name={name}
        value={value}
        onChange={onChange}
        required
      />

    </div>

  );
}


export default App;