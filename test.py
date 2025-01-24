import onnx

# Load the ONNX decoder model
onnx_decoder = onnx.load("trocr_decoder.onnx")

# Inspect inputs and outputs
print("Decoder Inputs:")
for input in onnx_decoder.graph.input:
    print(input.name)

print("\nDecoder Outputs:")
for output in onnx_decoder.graph.output:
    print(output.name)

