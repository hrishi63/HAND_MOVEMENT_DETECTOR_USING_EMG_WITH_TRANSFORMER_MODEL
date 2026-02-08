// ===============================
// EMG Envelope Streaming (ML Safe)
// BioAmp Candy / EXG Pill
// ===============================

#define SAMPLE_RATE 500
#define BAUD_RATE 115200
#define INPUT_PIN A0
#define BUFFER_SIZE 64

int circular_buffer[BUFFER_SIZE];
int data_index = 0;
long sum = 0;

void setup() {
  Serial.begin(BAUD_RATE);
}

void loop() {
  static unsigned long past = 0;
  unsigned long now = micros();
  unsigned long interval = now - past;
  past = now;

  static long timer = 0;
  timer -= interval;

  if (timer < 0) {
    timer += 1000000 / SAMPLE_RATE;

    int raw = analogRead(INPUT_PIN);
    int filtered = EMGFilter(raw);
    int envelope = getEnvelope(abs(filtered));

    // Send ONLY envelope for ML
    Serial.println(envelope);
  }
}

int getEnvelope(int abs_emg) {
  sum -= circular_buffer[data_index];
  sum += abs_emg;
  circular_buffer[data_index] = abs_emg;
  data_index = (data_index + 1) % BUFFER_SIZE;
  return (sum / BUFFER_SIZE) * 2;
}

// ===== Band-pass EMG filter =====
float EMGFilter(float input) {
  float output = input;

  {
    static float z1, z2;
    float x = output - 0.05159732*z1 - 0.36347401*z2;
    output = 0.01856301*x + 0.03712602*z1 + 0.01856301*z2;
    z2 = z1;
    z1 = x;
  }
  {
    static float z1, z2;
    float x = output - -0.53945795*z1 - 0.39764934*z2;
    output = x - 2*z1 + z2;
    z2 = z1;
    z1 = x;
  }
  {
    static float z1, z2;
    float x = output - 0.47319594*z1 - 0.70744137*z2;
    output = x + 2*z1 + z2;
    z2 = z1;
    z1 = x;
  }
  {
    static float z1, z2;
    float x = output - -1.00211112*z1 - 0.74520226*z2;
    output = x - 2*z1 + z2;
    z2 = z1;
    z1 = x;
  }
  return output;
}
