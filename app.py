import streamlit as st
import torch
import cv2
import numpy as np
from model_segmentation import get_model as get_seg_model
from model_classification import get_model as get_class_model
from PIL import Image
import time

# Page config
st.set_page_config(
    page_title="ASL Letter Recognition",
    page_icon="👋",
    layout="wide"
)

# Styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = (224, 224)
CLASSES = ['A', 'B', 'C', 'D', 'E']

@st.cache_resource
def load_models():
    """Load both classification and segmentation models"""
    try:
        # Load classification model
        class_model = get_class_model()
        class_model.load_state_dict(torch.load('best_model_classification.pth', map_location=DEVICE))
        class_model.to(DEVICE).eval()
        
        # Load segmentation model
        seg_model = get_seg_model()
        seg_model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
        seg_model.to(DEVICE).eval()
        
        return class_model, seg_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def preprocess_image(image):
    """Preprocess image for model input"""
    # Convert to RGB if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize and normalize
    image_resized = cv2.resize(image, IMG_SIZE)
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Convert to tensor
    image_tensor = torch.FloatTensor(image_normalized).permute(2, 0, 1).unsqueeze(0)
    return image_tensor.to(DEVICE), image_resized

def get_predictions(image_tensor, class_model, seg_model):
    """Get predictions from both models"""
    with torch.no_grad():
        # Classification prediction
        class_output = class_model(image_tensor)
        class_probs = torch.softmax(class_output, dim=1)[0]
        class_idx = class_probs.argmax().item()
        
        # Segmentation prediction
        seg_output = seg_model(image_tensor)
        seg_mask = torch.sigmoid(seg_output)[0, 0].cpu().numpy()
        
    return {
        'letter': CLASSES[class_idx],
        'confidence': class_probs[class_idx].item(),
        'mask': seg_mask > 0.5
    }

def main():
    st.title("ASL Letter Recognition (A-E)")
    
    # Load models
    class_model, seg_model = load_models()
    if class_model is None or seg_model is None:
        return
    
    # Sidebar
    st.sidebar.title("Options")
    input_mode = st.sidebar.radio("Select Input Mode", ["Upload Image", "Camera"])
    
    # Main content - split into columns
    col1, col2 = st.columns(2)
    
    if input_mode == "Upload Image":
        image_file = st.sidebar.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        if image_file:
            # Load and display original image
            image = Image.open(image_file)
            col1.header("Input Image")
            col1.image(image, use_container_width=True)
            
            # Process image and get predictions
            image_tensor, _ = preprocess_image(image)
            results = get_predictions(image_tensor, class_model, seg_model)
            
            # Display results
            col2.header("Results")
            col2.markdown(f"""
                ### Detected Letter: {results['letter']}
                ### Confidence: {results['confidence']:.2%}
            """)
            col2.image(results['mask'].astype(np.uint8) * 255, 
                      caption="Segmentation Mask",
                      use_container_width=True)
    
    else:  # Camera mode
        try:
            cap = cv2.VideoCapture(0)
            frame_placeholder = col1.empty()
            results_placeholder = col2.empty()
            
            stop_button = st.sidebar.button("Stop Camera")
            
            # Frame rate control
            last_time = time.time()
            fps_limit = 15  # Limit to 15 FPS
            
            while not stop_button:
                current_time = time.time()
                if current_time - last_time < 1.0 / fps_limit:
                    continue
                last_time = current_time
                
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)  # Mirror effect
                    frame_placeholder.image(frame, channels="BGR", use_container_width=True)
                    
                    # Process frame and get predictions
                    image_tensor, _ = preprocess_image(frame)
                    results = get_predictions(image_tensor, class_model, seg_model)
                    
                    # Update results
                    results_placeholder.markdown(f"""
                        ### Detected Letter: {results['letter']}
                        ### Confidence: {results['confidence']:.2%}
                    """)
                    results_placeholder.image(results['mask'].astype(np.uint8) * 255,
                                           caption="Segmentation Mask",
                                           use_container_width=True)
                    
                    # Print labels below the camera feed
                    col1.markdown(f"**Detected Letter:** {results['letter']}")
                    col1.markdown(f"**Confidence:** {results['confidence']:.2%}")
            
            cap.release()
        except Exception as e:
            st.error(f"Error accessing camera: {str(e)}")

if __name__ == "__main__":
    main()