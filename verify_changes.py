try:
    import torch
    from JGRM import JGRMModel
    import json

    print("Successfully imported JGRM modules.")

    # Load a config to check if it parses correctly
    config_path = '/home/shzheng2025/JGRM-Enc/config/chengdu.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    print("Successfully loaded config.")

    edge_index = [[0, 1], [1, 0]]

    model = JGRMModel(
        vocab_size=config['vocab_size'],
        route_max_len=config['route_max_len'],
        road_feat_num=config['road_feat_num'],
        road_embed_size=config['road_embed_size'],
        gps_feat_num=config['gps_feat_num'],
        gps_embed_size=config['gps_embed_size'],
        route_embed_size=config['route_embed_size'],
        hidden_size=config['hidden_size'],
        edge_index=edge_index,
        drop_edge_rate=config['drop_edge_rate'],
        drop_route_rate=config['drop_route_rate'],
        drop_road_rate=config['drop_road_rate'],
        use_vision=config.get('use_vision', False),
        use_vision_segment_encoder=config.get('use_vision_segment_encoder', False),
        vision_segment_window_size=config.get('vision_segment_window_size', 1),
        fusion_type=config.get('fusion_type', 'shared'),
        use_modality_embedding=config.get('use_modality_embedding', True),
        cross_modal_num_heads=config.get('cross_modal_num_heads', 4),
        cross_modal_num_layers=config.get('cross_modal_num_layers', 1),
        enable_stage2_fusion=config.get('enable_stage2_fusion', True),
    )

    print("Successfully instantiated JGRMModel.")
    print(f"use_vision: {model.use_vision}")
    print(f"use_vision_segment_encoder: {model.use_vision_segment_encoder}")
    print(f"vision_enabled: {model.vision_enabled}")

except Exception:
    import traceback
    traceback.print_exc()
    raise
