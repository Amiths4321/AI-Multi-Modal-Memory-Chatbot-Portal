import streamlit as st
import base64
from core.memory_manager import HybridMemoryManager
from core.llm_handler import QwenVLPipelineHandler

st.set_page_config(page_title="Qwen-VL Memory Engine", layout="wide", initial_sidebar_state="expanded")

# State Initializations
if "memory_engine" not in st.session_state:
    st.session_state.memory_engine = HybridMemoryManager()
if "vlm_engine" not in st.session_state:
    st.session_state.vlm_engine = QwenVLPipelineHandler()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "latest_recalled" not in st.session_state:
    st.session_state.latest_recalled = []

st.markdown("""
    <style>
        .stChatMessage { border-radius: 10px; margin-bottom: 10px; }
        .memory-card { background-color: #1E293B; border-left: 4px solid #8B5CF6; padding: 10px; margin-bottom: 8px; border-radius: 4px; font-size: 0.85rem;}
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR: Live Memory Tracking Hub ---
with st.sidebar:
    st.title("🧠 Vision Memory Vault")
    st.caption("Multimodal semantic vector pipeline logs")
    st.write("---")
    
    # Image Input Widget for Qwen-VL input mapping
    st.subheader("📸 Upload Vision Input")
    uploaded_image = st.file_uploader("Attach an image to pass to Qwen-VL", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        st.image(uploaded_image, caption="Staged Visual Matrix Context", use_column_width=True)
        
    st.write("---")
    st.subheader("💡 Recalled Text Memories")
    if st.session_state.latest_recalled:
        for memory in st.session_state.latest_recalled:
            st.markdown(f"<div class='memory-card'>{memory}</div>", unsafe_allow_html=True)
    else:
        st.info("Awaiting interactive loops...")

# --- MAIN VIEWPORT ---
st.title("⚡ Remote Qwen-VL Cognitive System")
st.subheader("Air-gapped architecture processing text + visions over remote high-compute nodes")

# Render active working message arrays
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Prompt Execution Hook
if user_prompt := st.chat_input("Ask Qwen-VL anything about text history or the attached image..."):
    
    # Render user query text
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    
    # Process image conversion if present
    img_b64 = None
    if uploaded_image:
        img_b64 = base64.b64encode(uploaded_image.read()).decode('utf-8')
    
    # Execute semantic data checks from local memory index
    with st.spinner("Retrieving historical memory context vectors..."):
        recalled_memories = st.session_state.memory_engine.retrieve_relevant_memories(user_prompt)
        st.session_state.latest_recalled = recalled_memories
    
    # Dispatch inference payload to your remote server machine 
    with st.chat_message("assistant"):
        with st.spinner("Remote GPU processing via Qwen-VL inference..."):
            response = st.session_state.vlm_engine.generate_response(
                current_message=user_prompt,
                short_term_history=st.session_state.chat_history[:-1],
                long_term_memories=recalled_memories,
                image_b64=img_b64
            )
        st.markdown(response)
        
    # Append state vectors and sync
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.session_state.memory_engine.save_memory(user_prompt, response)
    
    st.rerun()