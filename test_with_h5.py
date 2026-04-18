import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from lime import lime_image
from skimage.segmentation import mark_boundaries

# FIX 1: The NpEncoder handles the "Object of type float32 is not JSON serializable" error
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def main():
    # Setup paths
    model_path = 'models/cancer_classifier_xai.h5'
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    # Load Model
    model = load_model(model_path)
    print("Model loaded successfully.")

    # Load a test image (using the first one found in the dataset)
    # This is a simplified version of your data loading logic
    test_image = np.random.random((1, 224, 224, 3)) # Placeholder for your actual image loading
    
    # Predict
    preds = model.predict(test_image)
    pred_class = np.argmax(preds[0])
    
    # XAI with LIME
    explainer = lime_image.LimeImageExplainer()
    
    # FIX 2: Speed Hack - Reduced num_samples to 100 for fast results
    print("Generating LIME explanation (this will take ~1 minute)...")
    explanation = explainer.explain_instance(
        test_image[0].astype('double'), 
        model.predict, 
        top_labels=5, 
        hide_color=0, 
        num_samples=100 
    )

    # Visualization
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=True, 
        num_features=5, 
        hide_rest=False
    )
    
    plt.figure(figsize=(10, 5))
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.title(f"XAI Prediction: Class {pred_class}")
    
    # FIX 3: Save the image permanently before the session ends
    image_save_path = os.path.join(output_dir, 'xai_results.png')
    plt.savefig(image_save_path)
    print(f"Success! Heatmap saved to {image_save_path}")

    # Save JSON data
    xai_results = {
        "predicted_class": int(pred_class),
        "confidence": float(np.max(preds[0]) * 100),
        "status": "Inference Complete"
    }
    
    json_path = os.path.join(output_dir, 'xai_output.json')
    with open(json_path, 'w') as f:
        json.dump(xai_results, f, indent=4, cls=NpEncoder)
    print(f"Success! JSON data saved to {json_path}")

if __name__ == "__main__":
    main()
