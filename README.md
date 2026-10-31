import streamlit as st

# ====================
# 頁面設定
# ====================

st.set_page_config(
    page_title="Bloom Forest 芬芳森林",
    page_icon="🌳",
    layout="wide"
)

# ====================
# 初始化狀態
# ====================

if "confidence" not in st.session_state:
    st.session_state.confidence = 0

if "joy" not in st.session_state:
    st.session_state.joy = 0

if "hope" not in st.session_state:
    st.session_state.hope = 0

if "convenience" not in st.session_state:
    st.session_state.convenience = 0

# ====================
# 標題
# ====================

st.title("🌳 Bloom Forest 芬芳森林")

st.markdown("""
### 三好 → AI生成 → 四給影響力 → 歡喜布施

在芬芳森林裡，

🌱 存好心  
🌸 說好話  
🍃 做好事  

讓每一份善意，都化成一朵花。
""")

# ====================
# 分頁
# ====================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🌱 存好心",
        "🌸 說好話",
        "🍃 做好事",
        "✨ 四給花園",
        "💝 歡喜布施",
    ]
)

# ====================
# 存好心
# ====================

with tab1:

    st.header("🌱 存好心")

    gratitude = st.text_area(
        "今天想感謝什麼？",
        placeholder="例如：朋友陪我聊天"
    )

    if st.button("生成感恩花語"):

        if gratitude:

            flower = f"""
🌼 感恩花語

今天感謝 {gratitude}。

每一次真誠的陪伴，
都讓生命更加溫暖。

願這份善意持續盛開。
"""

            st.success(flower)

            st.session_state.joy += 1
            st.session_state.hope += 1

# ====================
# 說好話
# ====================

with tab2:

    st.header("🌸 說好話")

    situation = st.text_input(
        "想鼓勵的人或情境",
        placeholder="朋友考試失利"
    )

    if st.button("生成鼓勵花語"):

        if situation:

            flower = f"""
🌷 鼓勵花

給 {situation}

一次失敗不代表全部，
請相信自己的努力。

未來終將開出屬於自己的花。
"""

            st.success(flower)

            st.session_state.confidence += 1
            st.session_state.joy += 1

# ====================
# 做好事
# ====================

with tab3:

    st.header("🍃 做好事")

    if st.button("推薦今天的善意任務"):

        tasks = [
            "分享筆記給同學",
            "鼓勵一位朋友",
            "幫助長輩提重物",
            "撿拾路邊垃圾",
            "捐出一本二手書"
        ]

        st.success("🌻 今日善意任務")

        for task in tasks:
            st.write("✅", task)

    action = st.text_input(
        "今天做了什麼好事？"
    )

    if st.button("提交善行"):

        if action:

            st.success("🌻 善意已成功種下！")

            st.session_state.hope += 1
            st.session_state.convenience += 1

# ====================
# 四給花園
# ====================

with tab4:

    st.header("✨ 四給花園")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "🌟 給人信心",
            st.session_state.confidence
        )

        st.metric(
            "😊 給人歡喜",
            st.session_state.joy
        )

    with
