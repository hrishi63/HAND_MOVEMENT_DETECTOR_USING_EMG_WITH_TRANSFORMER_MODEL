import React, { useState } from "react";

export default function PatientInfoModal({ onStart }) {
  const [patientName, setPatientName] = useState("");
  const [age, setAge] = useState("");
  const [gender, setGender] = useState("");
  const [errors, setErrors] = useState({});

  const validateForm = () => {
    const newErrors = {};
    
    if (!patientName.trim()) {
      newErrors.name = "Patient name is required";
    }
    
    if (!age) {
      newErrors.age = "Age is required";
    } else if (age < 1 || age > 120) {
      newErrors.age = "Age must be between 1 and 120";
    }
    
    if (!gender) {
      newErrors.gender = "Gender is required";
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (validateForm()) {
      onStart({
        name: patientName.trim(),
        age: parseInt(age),
        gender: gender,
        timestamp: new Date().toISOString()
      });
    }
  };

  return (
    <div style={styles.overlay}>
      <div style={styles.modal}>
        <h2 style={styles.title}>🏥 Patient Information</h2>
        <p style={styles.subtitle}>Please enter patient details to begin monitoring</p>
        
        <form onSubmit={handleSubmit} style={styles.form}>
          {/* Patient Name */}
          <div style={styles.formGroup}>
            <label style={styles.label}>Patient Name *</label>
            <input
              type="text"
              value={patientName}
              onChange={(e) => setPatientName(e.target.value)}
              placeholder="Enter full name"
              style={{
                ...styles.input,
                borderColor: errors.name ? '#ff1744' : '#333'
              }}
            />
            {errors.name && <span style={styles.error}>{errors.name}</span>}
          </div>

          {/* Age */}
          <div style={styles.formGroup}>
            <label style={styles.label}>Age *</label>
            <input
              type="number"
              value={age}
              onChange={(e) => setAge(e.target.value)}
              placeholder="Enter age"
              min="1"
              max="120"
              style={{
                ...styles.input,
                borderColor: errors.age ? '#ff1744' : '#333'
              }}
            />
            {errors.age && <span style={styles.error}>{errors.age}</span>}
          </div>

          {/* Gender */}
          <div style={styles.formGroup}>
            <label style={styles.label}>Gender *</label>
            <div style={styles.radioGroup}>
              <label style={styles.radioLabel}>
                <input
                  type="radio"
                  value="Male"
                  checked={gender === "Male"}
                  onChange={(e) => setGender(e.target.value)}
                  style={styles.radio}
                />
                <span style={styles.radioText}>Male</span>
              </label>
              
              <label style={styles.radioLabel}>
                <input
                  type="radio"
                  value="Female"
                  checked={gender === "Female"}
                  onChange={(e) => setGender(e.target.value)}
                  style={styles.radio}
                />
                <span style={styles.radioText}>Female</span>
              </label>
              
              <label style={styles.radioLabel}>
                <input
                  type="radio"
                  value="Other"
                  checked={gender === "Other"}
                  onChange={(e) => setGender(e.target.value)}
                  style={styles.radio}
                />
                <span style={styles.radioText}>Other</span>
              </label>
            </div>
            {errors.gender && <span style={styles.error}>{errors.gender}</span>}
          </div>

          {/* Submit Button */}
          <button type="submit" style={styles.startButton}>
            Start Monitoring ▶️
          </button>
        </form>
      </div>
    </div>
  );
}

const styles = {
  overlay: {
    position: "fixed",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "rgba(0, 0, 0, 0.9)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 9999,
  },
  modal: {
    background: "linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)",
    borderRadius: "1rem",
    padding: "2.5rem",
    maxWidth: "500px",
    width: "90%",
    border: "2px solid #00e676",
    boxShadow: "0 0 40px rgba(0, 230, 118, 0.3)",
  },
  title: {
    fontSize: "2rem",
    marginBottom: "0.5rem",
    textAlign: "center",
    color: "#00e676",
    fontWeight: "bold",
  },
  subtitle: {
    textAlign: "center",
    color: "#888",
    marginBottom: "2rem",
    fontSize: "0.9rem",
  },
  form: {
    display: "flex",
    flexDirection: "column",
    gap: "1.5rem",
  },
  formGroup: {
    display: "flex",
    flexDirection: "column",
    gap: "0.5rem",
  },
  label: {
    color: "#e5e7eb",
    fontSize: "0.9rem",
    fontWeight: "600",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
  },
  input: {
    padding: "0.75rem",
    backgroundColor: "#0a0a0a",
    border: "2px solid #333",
    borderRadius: "0.5rem",
    color: "#fff",
    fontSize: "1rem",
    outline: "none",
    transition: "border-color 0.3s",
  },
  radioGroup: {
    display: "flex",
    gap: "1.5rem",
    marginTop: "0.5rem",
  },
  radioLabel: {
    display: "flex",
    alignItems: "center",
    gap: "0.5rem",
    cursor: "pointer",
  },
  radio: {
    width: "18px",
    height: "18px",
    cursor: "pointer",
    accentColor: "#00e676",
  },
  radioText: {
    color: "#e5e7eb",
    fontSize: "1rem",
  },
  error: {
    color: "#ff1744",
    fontSize: "0.8rem",
    marginTop: "0.25rem",
  },
  startButton: {
    marginTop: "1rem",
    padding: "1rem",
    backgroundColor: "#00e676",
    color: "#000",
    border: "none",
    borderRadius: "0.5rem",
    fontSize: "1.1rem",
    fontWeight: "bold",
    cursor: "pointer",
    transition: "all 0.3s",
    boxShadow: "0 0 20px rgba(0, 230, 118, 0.5)",
  },
};
