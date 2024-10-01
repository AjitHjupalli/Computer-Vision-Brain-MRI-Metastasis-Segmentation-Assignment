from sklearn.metrics import jaccard_score
from models.nested_unet import nested_unet

def assess_model(model, images, masks):
    """Evaluate model performance using DICE Score."""
    predictions = model.predict(images)
    dice_score = jaccard_score(masks.flatten(), predictions.flatten(), average='binary')
    return dice_score
