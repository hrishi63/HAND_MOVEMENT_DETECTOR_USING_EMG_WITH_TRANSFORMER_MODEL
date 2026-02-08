import React, { useState, useEffect, useRef } from "react";
import AIAnalyzer from "./AIAnalyzer";

export default function EMGHandVisualizer() {
  const [handState, setHandState] = useState("open");
  const [rms, setRms] = useState(0);
  const [history, setHistory] = useState(Array(140).fill(0));
  const [running, setRunning] = useState(true);
  const canvasRef = useRef(null);

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
    <div style={styles.container}>
      <h1 style={styles.title}>
        ⚡ EMG LIVE AI DASHBOARD
      </h1>
      
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
          style={styles.playerButton}
          onClick={() => setRunning(!running)}
        >
          {running ? "⏸" : "▶"}
        </button>
      </div>
      
      {/* ✅ AI ANALYZER */}
      <AIAnalyzer 
        handState={handState}
        running={running}
      />
      
      <p style={styles.footer}>
        Real Transformer · WebGPU Acceleration · Clinical Gesture AI
      </p>
    </div>
  );
}

/* ================= STYLES ================= */
const styles = {
  container: {
    minHeight: "100vh",
    background: "#000",
    color: "#e5e7eb",
    padding: "2rem",
    fontFamily: "Inter, system-ui"
  },
  title: {
    textAlign: "center",
    fontSize: "3rem",
    marginBottom: "2rem",
    color: "#fff",
    letterSpacing: "0.15em"
  },
  top: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr 1fr",
    gap: "1.5rem",
    maxWidth: "1200px",
    margin: "0 auto"
  },
  card: {
    background: "#050505",
    borderRadius: "1rem",
    padding: "1.5rem",
    border: "1px solid #111"
  },
  hand: {
    fontSize: "6rem",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "#050505",
    borderRadius: "1rem"
  },
  handEmoji: {
    animation: "scale 1.5s infinite alternate"
  },
  canvas: {
    marginTop: "2rem",
    width: "100%",
    background: "#000",
    borderRadius: "1rem"
  },
  controls: {
    marginTop: "2rem",
    display: "flex",
    justifyContent: "center"
  },
  playerButton: {
    width: "80px",
    height: "80px",
    fontSize: "32px",
    borderRadius: "50%",
    border: "none",
    background: "#111",
    color: "#00e676",
    boxShadow: "0 0 30px #00e67655",
    cursor: "pointer"
  },
  footer: {
    marginTop: "2rem",
    textAlign: "center",
    color: "#666"
  }
};
