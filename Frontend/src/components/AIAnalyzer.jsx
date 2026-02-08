import React, { useEffect, useRef, useState } from "react";

const WINDOW_SIZE = 256;

export default function AIAnalyzer({ handState, running }) {
  const [cycles, setCycles] = useState(0);
  const [healthStatus, setHealthStatus] = useState("Monitoring...");
  const [activation, setActivation] = useState(0);
  const [stability, setStability] = useState(0);
  const [sharpness, setSharpness] = useState(0);
  const [consistency, setConsistency] = useState(0);
  const [responsiveness, setResponsiveness] = useState(0);
  const [gestureConfidence, setGestureConfidence] = useState(0);
  const [debugInfo, setDebugInfo] = useState("Loading...");
  
  const sessionRef = useRef(null);
  const lastStateRef = useRef("open");
  const modelLoadedRef = useRef(false);
  const confidenceHistoryRef = useRef([]);

  /* ================= MODEL LOAD ================= */
  useEffect(() => {
    async function load() {
      if (sessionRef.current) return;
      try {
        console.log("🔄 Loading ONNX Runtime...");
        setDebugInfo("Loading ONNX Runtime...");
        
        const ort = await import("onnxruntime-web");
        
        console.log("🔄 Loading model files...");
        setDebugInfo("Loading model files...");
        
        // Fetch both files manually
        const modelResponse = await fetch("/model/emg_transformer_web.onnx");
        const modelArrayBuffer = await modelResponse.arrayBuffer();
        
        const dataResponse = await fetch("/model/emg_transformer_web.onnx.data");
        const dataArrayBuffer = await dataResponse.arrayBuffer();
        
        console.log("✅ Files fetched, creating session...");
        setDebugInfo("Creating AI session...");
        
        // Create session with external data
        sessionRef.current = await ort.InferenceSession.create(
          modelArrayBuffer,
          {
            executionProviders: ["wasm"],
            externalData: [
              {
                data: dataArrayBuffer,
                path: "emg_transformer_web.onnx.data"
              }
            ]
          }
        );
        
        modelLoadedRef.current = true;
        setDebugInfo("✅ AI Ready");
        console.log("✅ Model loaded successfully");
        
      } catch (e) {
        console.error("❌ Model load error:", e);
        setDebugInfo(`Error: ${e.message}`);
      }
    }
    load();
  }, []);

  /* ================= GESTURE CYCLES ================= */
  useEffect(() => {
    if (!handState) return;
    if (lastStateRef.current === "open" && handState === "clenched") {
      setCycles((prev) => {
        const val = prev + 1;
        if (val >= 5) {
          setHealthStatus("Healthy Muscle Response");
        }
        return val;
      });
    }
    lastStateRef.current = handState;
  }, [handState]);

  /* ================= AI INFERENCE ENGINE ================= */
  useEffect(() => {
    if (!modelLoadedRef.current || !running) {
      if (!running && modelLoadedRef.current) {
        setDebugInfo("⏸ Paused");
      }
      return;
    }

    const runAI = async () => {
      try {
        const res = await fetch("http://localhost:5000/ai_window");
        const data = await res.json();

        if (!data.ready || data.window.length !== WINDOW_SIZE) {
          setDebugInfo(`Buffering ${data.window?.length || 0}/256`);
          return;
        }

        const window = data.window;

        /* ===== Normalize ===== */
        const mean = window.reduce((a, b) => a + b, 0) / WINDOW_SIZE;
        let std = Math.sqrt(
          window.reduce((a, b) => a + (b - mean) ** 2, 0) / WINDOW_SIZE
        );
        if (std < 1e-6) std = 1.0;
        
        const norm = window.map((v) => (v - mean) / std);

        /* ===== Create tensor ===== */
        const ort = await import("onnxruntime-web");
        const tensor = new ort.Tensor(
          "float32",
          Float32Array.from(norm),
          [1, 256, 1]
        );

        /* ===== Run inference ===== */
        const results = await sessionRef.current.run({ input: tensor });
        const gestureProbability = results.output.data[0];

        console.log("🎯 AI:", gestureProbability.toFixed(4), "| State:", handState);

        /* ===== Gesture Confidence ===== */
        const confidenceValue = handState === "clenched" ? gestureProbability : (1 - gestureProbability);
        confidenceHistoryRef.current.push(confidenceValue);
        if (confidenceHistoryRef.current.length > 10) {
          confidenceHistoryRef.current.shift();
        }
        
        const avgConfidence = 
          confidenceHistoryRef.current.reduce((a, b) => a + b, 0) / 
          confidenceHistoryRef.current.length;

        setDebugInfo(`AI: ${(gestureProbability * 100).toFixed(1)}% | ${handState}`);

        /* ===== Compute metrics from signal ===== */
        const rawEnergy = window.reduce((a, b) => a + Math.abs(b), 0) / WINDOW_SIZE;
        const act = Math.min(100, (rawEnergy / 30) * 100);

        const cv = std / (Math.abs(mean) + 1e-6);
        const stab = Math.max(0, Math.min(100, 100 - cv * 50));

        const range = Math.max(...window) - Math.min(...window);
        const sharp = Math.min(100, (range / 100) * 100);

        const variance = std * std;
        const cons = variance > 0 
          ? Math.max(0, Math.min(100, 100 / (1 + variance / 10)))
          : 50;

        const diffs = window.slice(1).map((v, i) => Math.abs(v - window[i]));
        const meanDiff = diffs.reduce((a, b) => a + b) / diffs.length;
        const resp = Math.min(100, meanDiff * 20);

        /* ===== Update UI ===== */
        setGestureConfidence((avgConfidence * 100).toFixed(1));
        setActivation(act.toFixed(1));
        setStability(stab.toFixed(1));
        setSharpness(sharp.toFixed(1));
        setConsistency(cons.toFixed(1));
        setResponsiveness(resp.toFixed(1));

        /* ===== Health Status ===== */
        if (cycles >= 3 && avgConfidence > 0.7) {
          setHealthStatus("Excellent Muscle Control");
        } else if (cycles >= 3) {
          setHealthStatus("Good Muscle Response");
        }

      } catch (e) {
        console.error("❌ Inference error:", e);
        setDebugInfo(`Error: ${e.message}`);
      }
    };

    const id = setInterval(runAI, 400);
    return () => clearInterval(id);
  }, [running, handState, cycles]);

  /* ================= UI ================= */
  return (
    <div style={styles.panel}>
      <h2 style={styles.title}>🧠 AI ANALYZER</h2>
      
      <div style={styles.debug}>
        <p>{debugInfo}</p>
      </div>
      
      <div style={styles.grid}>
        <MetricCard label="AI Confidence" value={gestureConfidence} unit="%" highlight />
        <MetricCard label="Neural Activation" value={activation} unit="%" />
        <MetricCard label="Signal Stability" value={stability} unit="%" />
        <MetricCard label="Response Sharpness" value={sharpness} unit="%" />
        <MetricCard label="Muscle Consistency" value={consistency} unit="%" />
        <MetricCard label="Responsiveness" value={responsiveness} unit="%" />
      </div>
      
      <div style={styles.status}>
        <p><strong>Gesture Cycles:</strong> {cycles}</p>
        <p><strong>Health:</strong> <span style={{color: '#00e676'}}>{healthStatus}</span></p>
      </div>
    </div>
  );
}

/* ================= METRIC CARD ================= */
function MetricCard({ label, value, unit, highlight }) {
  const percentage = parseFloat(value) || 0;
  const color = percentage > 70 ? '#00e676' : percentage > 40 ? '#ffa726' : '#ff1744';
  
  return (
    <div style={{...styles.card, ...(highlight && styles.highlight)}}>
      <p style={styles.label}>{label}</p>
      <h3 style={{ ...styles.value, color }}>
        {value}{unit}
      </h3>
      <div style={styles.barContainer}>
        <div style={{ ...styles.bar, width: `${percentage}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

/* ================= STYLES ================= */
const styles = {
  panel: {
    marginTop: "3rem",
    padding: "2rem",
    background: "#050505",
    borderRadius: "1rem",
    border: "1px solid #111",
    maxWidth: "1200px",
    margin: "3rem auto 0",
  },
  title: {
    fontSize: "2rem",
    marginBottom: "1.5rem",
    textAlign: "center",
    color: "#00e676",
    letterSpacing: "0.1em",
  },
  debug: {
    marginBottom: "1rem",
    padding: "0.75rem",
    background: "#1a0a00",
    borderRadius: "0.5rem",
    border: "1px solid #ff6b6b44",
    textAlign: "center",
    color: "#ffa500",
    fontFamily: "monospace",
    fontSize: "0.9rem",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
    gap: "1rem",
    marginBottom: "2rem",
  },
  card: {
    background: "#0a0a0a",
    padding: "1.5rem",
    borderRadius: "0.5rem",
    border: "1px solid #1a1a1a",
  },
  highlight: {
    border: "2px solid #00e676",
    boxShadow: "0 0 15px rgba(0, 230, 118, 0.3)",
  },
  label: {
    fontSize: "0.85rem",
    color: "#888",
    marginBottom: "0.5rem",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
  },
  value: {
    fontSize: "2rem",
    fontWeight: "bold",
    marginBottom: "0.5rem",
  },
  barContainer: {
    width: "100%",
    height: "4px",
    background: "#1a1a1a",
    borderRadius: "2px",
    overflow: "hidden",
  },
  bar: {
    height: "100%",
    transition: "width 0.3s ease",
  },
  status: {
    textAlign: "center",
    padding: "1rem",
    background: "#0a0a0a",
    borderRadius: "0.5rem",
    border: "1px solid #1a1a1a",
  },
};
