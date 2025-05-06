from openai import OpenAI  # Thư viện OpenAI API
import streamlit as st
import os  # Làm việc với hệ thống file

st.title("My Chatbot")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"
client = OpenAI(api_key=".....................")  # Tạo một client từ API

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Định nghĩa đường dẫn lưu lịch sử chat
CHAT_HISTORY_FILE = "chat_history.txt"

# Tải lịch sử chat từ file TXT
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as file:
            lines = file.readlines()
            messages = []
            for i in range(0, len(lines), 2):  # Mỗi tin nhắn gồm 2 dòng: role + nội dung
                role = lines[i].strip()
                content = lines[i + 2].strip()
                messages.append({"role": role, "content": content})
            return messages
    return []

# Lưu lịch sử chat vào file TXT
def save_chat_history(messages):
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as file:
        for msg in messages:
            file.write(f"{msg['role']}\n{msg['content']}\n")

# Khởi tạo hoặc tải lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# Sidebar với nút xóa lịch sử chat và hiển thị lịch sử chat
with st.sidebar:
    st.header("History")
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        save_chat_history([])
        st.rerun()  # Tải lại ứng dụng để cập nhật giao diện

    # Hiển thị lịch sử chat trong sidebar
    if st.session_state.messages:
        for message in st.session_state.messages:
            if message["role"] == "user":  # Chỉ hiển thị câu hỏi từ người dùng
                st.write(f"{message['content']}")
    else:
        st.write("")

# Hiển thị lịch sử chat trong giao diện chính
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Giao diện chính của chatbot
if prompt := st.chat_input("How can I help?"):  # Khung nhập prompt
    st.session_state.messages.append({"role": "user", "content": prompt})  # Lưu prompt vào list messages
    with st.chat_message("user", avatar=USER_AVATAR):  # Hiển thị lên giao diện cùng với avatar tương ứng
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""
        for response in client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=st.session_state["messages"],
            stream=True,  # Nhận dữ liệu để hiển thị
        ):
            full_response += response.choices[0].delta.content or ""
            message_placeholder.markdown(full_response + "|")  # Hiển thị phản hồi
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Lưu lịch sử chat sau mỗi lần tương tác
save_chat_history(st.session_state.messages)
