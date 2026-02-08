import torch
from model import EMGClassifier

model = EMGClassifier()
model.load_state_dict(torch.load("emg_transformer_real.pth", map_location="cpu"))
model.eval()

print("✅ Model loaded")

dummy = torch.randn(1, 256, 1)

torch.onnx.export(
    model,
    dummy,
    "emg_transformer_web.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    do_constant_folding=True,
    export_params=True,
    dynamic_axes=None
)

print("✅ CLEAN WEB ONNX EXPORTED")

# Check if .data file was created
import os
if os.path.exists("emg_transformer_web.onnx.data"):
    print("⚠️ .data file created - need to copy both files")
else:
    print("✅ Single file export")

# Test it
try:
    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession("emg_transformer_web.onnx")
    test = np.random.randn(1, 256, 1).astype(np.float32)
    result = sess.run(None, {"input": test})
    print(f"✅ Test passed")
    print(f"   Output shape: {result[0].shape}")
    print(f"   Output value: {result[0][0][0]:.4f} (should be 0-1)")
except Exception as e:
    print(f"❌ Test failed: {e}")