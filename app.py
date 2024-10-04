import streamlit as st
from utils import pose_detection, agent2
from langgraph.graph import Graph
import time

# Initialize graph in session state if not already done
if "workflow" not in st.session_state:
    workflow = Graph()

    # Add nodes only once
    workflow.add_node("pose", pose_detection)
    workflow.add_node("agent2", agent2)

    # Create edge and set entry/finish points
    workflow.add_edge("pose", "agent2")
    workflow.set_entry_point("pose")
    workflow.set_finish_point("agent2")

    # Store the compiled graph in session state
    st.session_state.workflow = workflow.compile()

# Streamlit app main function
def main():
    st.title("Sleep Position Detection & Classification")

    # Step 1: Upload image
    uploaded_file = st.file_uploader("Upload an image for pose detection", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded image to a temporary location
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.read())

        # Display the uploaded image
        st.image("temp_image.jpg", caption="Uploaded Image", use_column_width=True)

        # Initialize progress logging
        progress_text = st.empty()
        progress_bar = st.progress(0)

        # Step 2: Execute the workflow
        input_data = {"image_path": "temp_image.jpg"}

        # Logging the step - Pose Detection
        progress_text.write("Running Pose Detection...")
        progress_bar.progress(33)
        time.sleep(1)  # Simulating delay for pose detection
        result_pose = st.session_state.workflow.invoke(input_data)  # Call pose detection

        # Logging the step - Classification
        progress_text.write("Running Classification...")
        progress_bar.progress(66)
        time.sleep(1)  # Simulating delay for classification
        result_classification = st.session_state.workflow.invoke(input_data)  # Call classification

        # Final log - Completed
        progress_text.write("Completed.")
        progress_bar.progress(100)

        # Step 3: Display the result
        st.write(result_classification)

if __name__ == "__main__":
    main()


