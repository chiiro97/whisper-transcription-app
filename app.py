import streamlit as st
import whisper
import tempfile
import os

# Whisperモデルをキャッシュして一度だけロード
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

st.title("🎧 音声アップロード → 自動文字起こしアプリ")

# 音声ファイルのアップロード
uploaded_file = st.file_uploader(
    "音声ファイルをアップロードしてください (対応形式: wav, mp3, m4a, flac, ogg)",
    type=["wav", "mp3", "m4a", "flac", "ogg"]
)

if uploaded_file is not None:
    # 一時ファイルに保存（Whisperはファイルパスを必要とする）
    with tempfile.NamedTemporaryFile(suffix="."+uploaded_file.name.split(".")[-1], delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_filepath = tmp_file.name

    # 音声を再生可能に表示
    st.audio(uploaded_file, format='audio/wav')

    # 文字起こし実行
    with st.spinner("📝 文字起こし中...少々お待ちください"):
        try:
            # 日本語指定（日本語以外も対応したい場合は language 指定を省略）
            result = model.transcribe(tmp_filepath, language="ja")
            text = result["text"]
            st.success("✅ 文字起こし完了！")

            st.subheader("📄 文字起こし結果")
            st.text_area("テキスト", value=text, height=200)

        except Exception as e:
            st.error(f"⚠️ 文字起こし中にエラーが発生しました: {e}")

        finally:
            # 一時ファイルを削除
            if os.path.exists(tmp_filepath):
                os.remove(tmp_filepath)
