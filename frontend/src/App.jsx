
import { useState, useRef } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [type, setType] = useState("image");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Reference to file input
  const fileInputRef = useRef(null);

  // Handle media selection
  const handleFile = (event) => {
    const selected = event.target.files[0];

    if (!selected) return;

    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
    setError("");
  };

  // Switch between image and video
  const handleTypeChange = (newType) => {
    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setType(newType);
    setFile(null);
    setPreview(null);
    setResult(null);
    setError("");

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // Clear selected media and detection result
  const handleClear = () => {
    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setFile(null);
    setPreview(null);
    setResult(null);
    setError("");

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // Send media to FastAPI
  const handleDetect = async () => {
    if (!file) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const endpoint =
        type === "image"
          ? "http://127.0.0.1:8000/predict/image"
          : "http://127.0.0.1:8000/predict/video";

      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Detection failed");
      }

      setResult(data);

    } catch (err) {
      setError(
        err.message || "Unable to connect to backend."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">

      {/* HEADER */}
      <header className="navbar">
        <h2>Deepfake Detection System</h2>
        {/* <span>AI Powered Detection</span> */}
      </header>

      <main className="dashboard-main">

        {/* PAGE HEADING */}
        <section className="hero">
          <h1>Detect Fake or Real Media</h1>
          <p>
            Analyze images and videos using your
            EfficientNet-B0 deep learning model.
          </p>
        </section>

        {/* TWO COLUMN DASHBOARD */}
        <div className="dashboard-grid">

          {/* LEFT SIDE - MEDIA UPLOAD */}
          <section className="upload-card">

            <h2>Upload Your Media</h2>

            <div className="tabs">
              <button
                className={type === "image" ? "active" : ""}
                onClick={() => handleTypeChange("image")}
                disabled={loading}
              >
                Image Detection
              </button>

              <button
                className={type === "video" ? "active" : ""}
                onClick={() => handleTypeChange("video")}
                disabled={loading}
              >
                Video Detection
              </button>
            </div>

            {/* FILE UPLOAD */}
            <label className="upload-area">
              <input
                ref={fileInputRef}
                type="file"
                accept={
                  type === "image" ? "image/*" : "video/*"
                }
                onChange={handleFile}
                hidden
                disabled={loading}
              />

              <div className="upload-icon">↑</div>
              <h3>Select {type}</h3>
              <p>Click here to browse your files</p>
            </label>

            {/* MEDIA PREVIEW */}
            {preview && (
              <div className="preview">
                <h3>Selected Media</h3>

                {type === "image" ? (
                  <img
                    src={preview}
                    alt="Selected preview"
                  />
                ) : (
                  <video src={preview} controls />
                )}

                <p>{file?.name}</p>
              </div>
            )}

            {/* DETECT AND CLEAR BUTTONS */}
            <div className="action-buttons">

              <button
                className="detect-button"
                disabled={!file || loading}
                onClick={handleDetect}
              >
                {loading ? (
                  <span className="loading-content">
                    <span className="spinner"></span>
                    Analyzing Media...
                  </span>
                ) : (
                  `Detect ${type === "image" ? "Image" : "Video"}`
                )}
              </button>

              <button
                className="clear-button"
                onClick={handleClear}
                disabled={
                  loading || (!file && !result && !error && !preview)
                }
              >
                Clear
              </button>

            </div>

            {/* ERROR MESSAGE */}
            {error && (
              <div className="error-result">
                {error}
              </div>
            )}

          </section>

          {/* RIGHT SIDE - DETECTION RESULT */}
          <section className="result-panel">

            {!result && !loading && !error && (
              <div className="empty-result">
                <div className="empty-icon">⌁</div>
                <h2>Detection Results</h2>
                <p>
                  Upload an image or video and click Detect
                  to see the prediction here.
                </p>
              </div>
            )}

            {loading && (
              <div className="empty-result">
                <span className="large-spinner"></span>
                <h2>Analyzing Media</h2>
                <p>
                  Your media is being processed by the
                  EfficientNet-B0 model.
                </p>
              </div>
            )}

            {error && (
              <div className="empty-result">
                <h2>Detection Failed</h2>
                <p>{error}</p>
              </div>
            )}

            {result && (
              <div className="result-card">

                <h2>Detection Report</h2>

                <div
                  className={`prediction-badge ${
                    result.prediction === "REAL"
                      ? "real"
                      : "fake"
                  }`}
                >
                  {result.prediction === "REAL"
                    ? "✓ REAL"
                    : "✕ FAKE"}
                </div>

                {/* CONFIDENCE */}
                <div className="confidence-section">
                  <p>Model Confidence</p>

                  <h1>{result.confidence}%</h1>

                  <div className="progress-track">
                    <div
                      className="confidence-progress"
                      style={{
                        width: `${result.confidence}%`,
                      }}
                    />
                  </div>
                </div>

                {/* PROBABILITIES */}
                <div className="probability-section">

                  <div className="probability-heading">
                    <span>Fake Probability</span>
                    <strong>
                      {result.fake_probability}%
                    </strong>
                  </div>

                  <div className="progress-track">
                    <div
                      className="fake-progress"
                      style={{
                        width: `${result.fake_probability}%`,
                      }}
                    />
                  </div>

                  <div className="probability-heading">
                    <span>Real Probability</span>
                    <strong>
                      {result.real_probability}%
                    </strong>
                  </div>

                  <div className="progress-track">
                    <div
                      className="real-progress"
                      style={{
                        width: `${result.real_probability}%`,
                      }}
                    />
                  </div>

                </div>

                {/* VIDEO DETAILS */}
                {type === "video" && (
                  <div className="video-details">
                    <h3>Video Analysis</h3>

                    <div className="video-stats">

                      <div>
                        <strong>
                          {result.frames_checked}
                        </strong>
                        <span>Frames Analyzed</span>
                      </div>

                      <div>
                        <strong>
                          {result.total_frames}
                        </strong>
                        <span>Total Frames</span>
                      </div>

                      <div>
                        <strong>
                          {result.duration_seconds}s
                        </strong>
                        <span>Duration</span>
                      </div>

                    </div>
                  </div>
                )}

              </div>
            )}

          </section>

        </div>
      </main>
    </div>
  );
}

export default App;
