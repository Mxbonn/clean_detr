def convert_original_state_dict(model_state_dict):
    """Convert state dict from original implementation to the current implementation."""
    new_state_dict = {}
    for k, v in model_state_dict.items():
        if k.startswith("backbone.0."):
            k = k.replace("backbone.0.", "backbone.backbone.")
        if k.startswith("transformer.encoder."):
            k = k.replace("transformer.encoder.", "backbone.transformer_encoder.")
        if k.startswith("input_proj."):
            k = k.replace("input_proj.", "backbone.input_proj.")
        if k.startswith("transformer.decoder."):
            k = k.replace("transformer.decoder.", "transformer_decoder.")
        if k == "class_embed.weight":
            k = "class_embed.0.weight"
        if k == "class_embed.bias":
            k = "class_embed.0.bias"
        if k.startswith("bbox_embed.layers."):
            _, _, layer_idx, rest = k.split(".", 3)
            k = f"bbox_embed.{str(int(layer_idx)*2)}.{rest}"
        new_state_dict[k] = v
    return new_state_dict
