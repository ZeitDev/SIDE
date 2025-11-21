import timm

def Encoder(model_name='efficientnet_b0', pretrained=True):
    encoder = timm.create_model(model_name, features_only=True, pretrained=pretrained)
    return encoder