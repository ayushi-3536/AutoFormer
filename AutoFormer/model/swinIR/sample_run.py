import torch

from AutoFormer.model.swinIR.network_swinir import SwinIR


def main():
    super_cfg = {
        'depths': [6, 6, 6, 6],
        'embed_dim': 60,
        'num_heads': [6, 6, 6, 6],
        'mlp_ratio': 2.,
    }

    same_as_super_cfg = {
            'rstb_num': 4,                   # Num of RSTB layers/blocks in the network
            'stl_num': 6,                    # Num STL blocks per RSTB layer
            'embed_dim': [60, 60, 60, 60],   # Per RSTB layer
            'mlp_ratio': [2., 2., 2., 2.], # Per RSTB layer
            'num_heads': [6, 6, 6, 6],       # Per RSTB layer
    }

    model = SwinIR(img_size=64,
                   window_size=8,
                   depths=super_cfg['depths'], 
                   embed_dim=super_cfg['embed_dim'], 
                   num_heads=super_cfg['num_heads'], 
                   mlp_ratio=super_cfg['mlp_ratio'],
                   upsampler='pixelshuffledirect',
                   upscale=4)
    print("model (requires_grad cnt): ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model.set_sample_config(same_as_super_cfg)

    # img = torch.rand(32, 3, 64, 64)
    # out = model(img)
    # print('Output shape: ', out.shape)
    params = model.get_sampled_params_numel(same_as_super_cfg)
    print("model (params_numel cnt): ", params)

    #--------------------------------
    # Check with smaller sample
    # Idea: The bigger `model` sampled with a small sample_cfg
    #       should have same params as a newly initialized `model_small`
    #       with same sample config values

    sample_cfg = {
            'rstb_num': 2,                   # Num of RSTB layers/blocks in the network
            'stl_num': 4,                    # Num STL blocks per RSTB layer
            'embed_dim': [48, 48],   # Per RSTB layer
            'mlp_ratio': [1.5, 1.5], # Per RSTB layer
            'num_heads': [4, 4],       # Per RSTB layer
    }

    model.set_sample_config(sample_cfg)
    params_sample = model.get_sampled_params_numel(sample_cfg)
    print("model with small-config (params_numel cnt): ", params_sample)

    model_small = SwinIR(img_size=64,
                        window_size=8,
                        depths=[4, 4],
                        embed_dim=48,
                        num_heads=[4, 4],
                        mlp_ratio=1.5,
                        upsampler='pixelshuffledirect',
                        upscale=4)
    print("model_small (requires_grad cnt): ", sum(p.numel() for p in model_small.parameters() if p.requires_grad))


if __name__ == '__main__':
    main()
