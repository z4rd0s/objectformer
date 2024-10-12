dependencies = ['torch']

def objectformer(pretrained=True, **kwargs):
    from objectformer import ObjectFormer
    model = ObjectFormer(**kwargs)
    if pretrained:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                'https://github.com/wdrink/Objectformer/releases/download/v0.1.0/objectformer.pth')
            )
    return 