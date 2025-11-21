import timm

def Encoder(encoder_name='efficientnet_b0', pretrained=True):
    encoder = timm.create_model(encoder_name, features_only=True, pretrained=pretrained)
    return encoder