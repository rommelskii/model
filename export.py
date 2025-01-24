import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

# Initialize the model and processor
model_name = "microsoft/trocr-base-handwritten"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
processor = TrOCRProcessor.from_pretrained(model_name, use_fast=True)

# Set the model to evaluation mode
model.eval()

# Example input: a dummy image tensor
# Replace this with your input shape (batch_size, channels, height, width)
batch_size = 1
dummy_image = torch.randn(batch_size, 3, 384, 384)  # For TrOCR, input images are typically resized to 384x384

# Dummy decoder input IDs for sequence-to-sequence models
dummy_decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long)  # Example: start token

# Set device for export
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
dummy_image = dummy_image.to(device)
dummy_decoder_input_ids = dummy_decoder_input_ids.to(device)

# Export the encoder (vision model) to ONNX
encoder_path = "trocr_encoder.onnx"
torch.onnx.export(
    model.encoder,
    (dummy_image,),
    encoder_path,
    input_names=["pixel_values"],
    output_names=["encoder_hidden_states"],
    dynamic_axes={"pixel_values": {0: "batch_size"}, "encoder_hidden_states": {0: "batch_size"}},
    opset_version=14,
)

print(f"Encoder exported to {encoder_path}")

# Export the decoder (text generation model) to ONNX
decoder_path = "trocr_decoder.onnx"
torch.onnx.export(
    model.decoder,
    (dummy_decoder_input_ids, model.encoder(dummy_image)[0]),
    decoder_path,
    input_names=["input_ids", "encoder_hidden_states"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"},
                  "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
                  "logits": {0: "batch_size", 1: "sequence_length"}},
    opset_version=13,
)

print(f"Decoder exported to {decoder_path}")

