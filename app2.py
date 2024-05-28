import streamlit as st

#ヘッダー表示
st.header("streamlitテストアプリ")
#タイトル表示
st.title("Webアプリ開発中")
#テキスト表示
st.write("生活用アプリ")

#スライダー（デフォルトでは0~100）
st.title("スライダー")
weight = st.slider("今日の体重は")
st.write("今の体重は" + str(weight) +"kgです")

#ボタン
st.title("今日の天気は")
st.button("リセット", type="primary")
if st.button("晴れ？"):
    st.write("今日も元気に！")
else:
    st.write("傘を忘れずに")

#テキスト入力
st.title("やること")
st.text_input("今やること", key="do")
st.session_state.do #keyでアクセス

#チェックボックス
st.title("ごみ捨てチェック")
is_agree = st.checkbox("ごみ捨てた？")
if is_agree:
    st.write("お疲れ様！")
else:
    st.write("忘れずに！")

