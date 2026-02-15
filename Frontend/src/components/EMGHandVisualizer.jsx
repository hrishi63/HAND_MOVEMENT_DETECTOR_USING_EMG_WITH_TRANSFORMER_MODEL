import React, { useState, useEffect, useRef } from "react";
import AIAnalyzer from "./AIAnalyzer";
import PatientInfoModal from "./PatientInfoModal";

export default function EMGHandVisualizer() {
  const [handState, setHandState] = useState("open");
  const [rms, setRms] = useState(0);
  const [history, setHistory] = useState(Array(140).fill(0));
  const [running, setRunning] = useState(false);
  const [recording, setRecording] = useState(false);
  const [patientInfo, setPatientInfo] = useState(null);
  const [showModal, setShowModal] = useState(true);
  
  const canvasRef = useRef(null);
  const recordedDataRef = useRef([]);

  /* ================= PATIENT INFO HANDLER ================= */
  const handlePatientStart = (info) => {
    setPatientInfo(info);
    setShowModal(false);
    setRunning(true);
    console.log("Patient info:", info);
  };

  const handleReset = () => {
    if (window.confirm("Are you sure you want to reset? All unsaved data will be lost.")) {
      setPatientInfo(null);
      setShowModal(true);
      setRunning(false);
      setRecording(false);
      recordedDataRef.current = [];
      setHistory(Array(140).fill(0));
    }
  };

  /* ================= DATA STREAM ================= */
  useEffect(() => {
    if (!running) return;
    
    const fetchData = async () => {
      try {
        const res = await fetch("http://localhost:5000/prediction");
        const data = await res.json();
        const rmsVal = Number(data.rms) || 0;
        const state = data.clenched ? "clenched" : "open";
        
        setHandState(state);
        setRms(rmsVal);
        setHistory(prev => [...prev.slice(1), rmsVal]);
      } catch {}
    };
    
    const id = setInterval(fetchData, 100);
    return () => clearInterval(id);
  }, [running]);

  /* ================= WAVEFORM ================= */
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const color = handState === "clenched" ? "#ff1744" : "#00e676";
    ctx.strokeStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 25;
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    history.forEach((v, i) => {
      const x = (i / history.length) * canvas.width;
      const y = canvas.height - Math.min(v / 120, 1) * canvas.height;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    
    ctx.stroke();
  }, [history, handState]);

  /* ================= UI ================= */
  const glow = handState === "clenched" ? "#ff1744" : "#00e676";

  return (
    <>
      {showModal && <PatientInfoModal onStart={handlePatientStart} />}
      
      <div style={styles.container}>
        <h1 style={styles.title}>⚡ EMG LIVE AI DASHBOARD</h1>
        
        {/* Patient Info Bar */}
        {patientInfo && (
          <div style={styles.patientBar}>
            <div style={styles.patientInfo}>
              <span style={styles.patientLabel}>Patient:</span>
              <span style={styles.patientValue}>{patientInfo.name}</span>
              <span style={styles.patientDivider}>|</span>
              <span style={styles.patientLabel}>Age:</span>
              <span style={styles.patientValue}>{patientInfo.age}</span>
              <span style={styles.patientDivider}>|</span>
              <span style={styles.patientLabel}>Gender:</span>
              <span style={styles.patientValue}>{patientInfo.gender}</span>
              <span style={styles.patientDivider}>|</span>
              <span style={styles.patientLabel}>Session:</span>
              <span style={styles.patientValue}>
                {new Date(patientInfo.timestamp).toLocaleString()}
              </span>
            </div>
            {recording && (
              <div style={styles.recordingIndicator}>
                <span style={styles.recordingDot}>●</span> Recording
              </div>
            )}
          </div>
        )}
        
        <div style={styles.top}>
          <div style={{ ...styles.card, boxShadow: `0 0 25px ${glow}` }}>
            <p>RMS SIGNAL</p>
            <h2>{rms.toFixed(2)} μV</h2>
          </div>
          
          <div style={styles.card}>
            <p>HAND STATE</p>
            <h2 style={{ color: glow }}>
              {handState.toUpperCase()}
            </h2>
          </div>
          
          <div style={{ ...styles.hand, boxShadow: `0 0 40px ${glow}` }}>
            <span style={styles.handEmoji}>
              {handState === "clenched" ? "✊" : "✋"}
            </span>
          </div>
        </div>
        
        <canvas
          ref={canvasRef}
          width={1200}
          height={220}
          style={styles.canvas}
        />
        
        <div style={styles.controls}>
          <button
            style={{
              ...styles.controlButton,
              backgroundColor: running ? "#ffa726" : "#00e676",
              color: running ? "#000" : "#000",
            }}
            onClick={() => setRunning(!running)}
          >
            {running ? "⏸ Pause" : "▶ Play"}
          </button>
          
          <button
            style={{
              ...styles.controlButton,
              backgroundColor: recording ? "#ff1744" : "#333",
              color: recording ? "#fff" : "#888",
            }}
            onClick={() => setRecording(!recording)}
            disabled={!running}
          >
            {recording ? "⏹ Stop Recording" : "⏺ Record"}
          </button>
          
          <button
            style={{
              ...styles.controlButton,
              backgroundColor: "#2196f3",
            }}
            onClick={() => {
              // Export function will be handled by AIAnalyzer
              const event = new CustomEvent("exportData");
              window.dispatchEvent(event);
            }}
            disabled={!patientInfo}
          >
            📥 Export Data
          </button>
          
          <button
            style={{
              ...styles.controlButton,
              backgroundColor: "#9c27b0",
            }}
            onClick={handleReset}
          >
            🔄 Reset Patient
          </button>
        </div>
        
        {/* AI ANALYZER */}
        <AIAnalyzer 
          handState={handState}
          running={running}
          recording={recording}
          patientInfo={patientInfo}
        />
        
        <p style={styles.footer}>
          Real Transformer · WebGPU Acceleration · Clinical Gesture AI
        </p>
      </div>
    </>
  );
}

/* ================= STYLES ================= */
const styles = {
  container: {
    minHeight: "100vh",
    background: "#000",
    color: "#e5e7eb",
    padding: "2rem",
    fontFamily: "Inter, system-ui",
  },
  title: {
    textAlign: "center",
    fontSize: "3rem",
    marginBottom: "1rem",
    color: "#fff",
    letterSpacing: "0.15em",
  },
  patientBar: {
    maxWidth: "1200px",
    margin: "0 auto 2rem",
    padding: "1rem 1.5rem",
    background: "linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)",
    borderRadius: "0.5rem",
    border: "1px solid #00e676",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  patientInfo: {
    display: "flex",
    gap: "0.75rem",
    alignItems: "center",
    flexWrap: "wrap",
  },
  patientLabel: {
    color: "#888",
    fontSize: "0.9rem",
    fontWeight: "600",
  },
  patientValue: {
    color: "#00e676",
    fontSize: "0.95rem",
    fontWeight: "bold",
  },
  patientDivider: {
    color: "#444",
  },
  recordingIndicator: {
    display: "flex",
    alignItems: "center",
    gap: "0.5rem",
    color: "#ff1744",
    fontWeight: "bold",
    fontSize: "0.9rem",
  },
  recordingDot: {
    fontSize: "1.5rem",
    animation: "blink 1s infinite",
  },
  top: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr 1fr",
    gap: "1.5rem",
    maxWidth: "1200px",
    margin: "0 auto",
  },
  card: {
    background: "#050505",
    borderRadius: "1rem",
    padding: "1.5rem",
    border: "1px solid #111",
  },
  hand: {
    fontSize: "6rem",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "#050505",
    borderRadius: "1rem",
  },
  handEmoji: {
    animation: "scale 1.5s infinite alternate",
  },
  canvas: {
    marginTop: "2rem",
    width: "100%",
    background: "#000",
    borderRadius: "1rem",
  },
  controls: {
    marginTop: "2rem",
    display: "flex",
    justifyContent: "center",
    gap: "1rem",
    flexWrap: "wrap",
  },
  controlButton: {
    padding: "1rem 2rem",
    fontSize: "1rem",
    fontWeight: "bold",
    borderRadius: "0.5rem",
    border: "none",
    cursor: "pointer",
    transition: "all 0.3s",
    boxShadow: "0 4px 6px rgba(0, 0, 0, 0.3)",
  },
  footer: {
    marginTop: "2rem",
    textAlign: "center",
    color: "#666",
  },
};
