from AutoFormer.model.vision_transformer.supernet_transformer import Vision_TransformerSuper


def main():
    super_cfg = {
        'embed_dim': 40,
        'depth': 4,
        'num_heads': 4,
        'mlp_ratio': 2.,
    }

    same_as_super_cfg = {
        'embed_dim': [40, 40, 40, 40],
        'layer_num': 4,
        'num_heads': [4, 4, 4, 4],
        'mlp_ratio': [2., 2., 2., 2.],
    }

    model = Vision_TransformerSuper(img_size=64,
                                    patch_size=8,
                                    embed_dim=super_cfg['embed_dim'], depth=super_cfg['depth'],
                                    num_heads=super_cfg['num_heads'], mlp_ratio=super_cfg['mlp_ratio'],
                                    qkv_bias=True, drop_rate=0.2,
                                    drop_path_rate=0.2,
                                    change_qkv=True)

    print("model (requires_grad cnt): ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model.set_sample_config(same_as_super_cfg)

    # img = torch.rand(32, 3, 64, 64)
    # out = model(img)
    # print('Output shape: ', out.shape)
    params = model.get_sampled_params_numel(same_as_super_cfg)
    print("model (params_numel cnt): ", params)

    # --------------------------------
    # Check with smaller sample
    # Idea: The bigger `model` sampled with a small sample_cfg
    #       should have same params as a newly initialized `model_small`
    #       with same sample config values

    new_sample_cfg = {
        'embed_dim': [20, 20],
        'layer_num': 2,
        'num_heads': [2, 2],
        'mlp_ratio': [1.5, 1.5],
    }

    model.set_sample_config(new_sample_cfg)
    params_sample = model.get_sampled_params_numel(new_sample_cfg)
    print("model with small-config (params_numel cnt): ", params_sample)

    model_small = Vision_TransformerSuper(img_size=64,
                                          patch_size=8,
                                          embed_dim=20, depth=2,
                                          num_heads=2, mlp_ratio=1.5,
                                          qkv_bias=True, drop_rate=0.2,
                                          drop_path_rate=0.2,
                                          change_qkv=True)

    print("model_small (requires_grad cnt): ", sum(p.numel() for p in model_small.parameters() if p.requires_grad))


if __name__ == '__main__':
    main()
