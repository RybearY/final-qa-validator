import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
import filetype
from soundfile import LibsndfileError
from pydub import AudioSegment
from copy import deepcopy
from io import BytesIO
import gc  # 가비지 컬렉션 추가

st.set_page_config(page_title="Audio Requirements Validator", layout="wide")

# 메모리 최적화를 위한 세션 상태 초기화 함수
def init_session_state():
    if "disabled" not in st.session_state:
        st.session_state.disabled = False
    if "start_button_clicked" not in st.session_state:
        st.session_state.start_button_clicked = False
    if "required_format" not in st.session_state:
        st.session_state.required_format = None
    if "required_channels" not in st.session_state:
        st.session_state.required_channels = None
    if "required_sample_rate" not in st.session_state:
        st.session_state.required_sample_rate = None
    if "required_bit_depth" not in st.session_state:
        st.session_state.required_bit_depth = None
    if "required_noise_floor" not in st.session_state:
        st.session_state.required_noise_floor = None
    if "required_stereo_status" not in st.session_state:
        st.session_state.required_stereo_status = None

# 파일 유효성 검사 함수
def validate_filetype(buffer):
    validate_mimetypes = [
        "audio/mpeg", "audio/mp4", "video/mp4", 
        "audio/x-wav", "audio/x-aiff", "audio/x-flac", "audio/ogg"
    ]
    kind = filetype.guess(buffer)
    if kind is None:
        return False, "The file format is not recognized."
    mime_type = kind.mime
    if mime_type not in validate_mimetypes:
        return False, f"파일 형식이 잘못되었거나, 지원하지 않는 형식입니다.", mime_type.split("/")[-1].upper()
    return True, "파일 분석 가능", mime_type.split("/")[-1].split('-')[-1].upper()

# 오디오 속성 분석 함수 (최적화)
def get_audio_properties_from_buffer(buffer):
    try:
        with sf.SoundFile(deepcopy(buffer), 'r') as f:
            try:
                # 전체 데이터를 읽지 않고 필요한 정보만 추출
                subtype = f.subtype
                if 'PCM_' in subtype:
                    bit_depth = subtype.replace('PCM_', '')
                else:
                    bit_depth = 'Unknown'
                samplerate = f.samplerate
                channels = f.channels
                duration = f.frames / samplerate

                return {
                    "Sample Rate": samplerate,
                    "Channels": channels,
                    "Bit Depth": bit_depth,
                    "Duration (seconds)": round(duration, 2)
                }
            except Exception as e:
                return {
                    "Sample Rate": "Error",
                    "Channels": "Error",
                    "Bit Depth": "Error",
                    "Duration (seconds)": "Error (Processing Failed)"
                }
    except LibsndfileError:
        try:
            wav_buffer = BytesIO()
            audio_file = AudioSegment.from_file(deepcopy(buffer))
            audio_file.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            
            with sf.SoundFile(wav_buffer, 'r') as f:
                subtype = f.subtype
                if 'PCM_' in subtype:
                    bit_depth = subtype.replace('PCM_', '')
                else:
                    bit_depth = 'Unknown'
                samplerate = f.samplerate
                channels = f.channels
                duration = f.frames / samplerate

                return {
                    "Sample Rate": samplerate,
                    "Channels": channels,
                    "Bit Depth": bit_depth,
                    "Duration (seconds)": round(duration, 2)
                }
        except Exception as e:
            return {
                "Sample Rate": "Error",
                "Channels": "Error",
                "Bit Depth": "Error",
                "Duration (seconds)": "Error (Processing Failed)"
            }

# 노이즈 플로어 계산 함수 (최적화 - 샘플링 사용)
def calculate_noise_floor_from_buffer(buffer, silence_threshold_db=-60):
    try:
        # 전체 파일을 읽지 않고 일부만 샘플링
        with sf.SoundFile(deepcopy(buffer), 'r') as f:
            # 파일의 처음, 중간, 끝 부분에서 각각 5초씩만 읽기
            total_frames = f.frames
            sample_size = min(int(f.samplerate * 5), total_frames // 3)
            
            # 처음 부분
            f.seek(0)
            start_data = f.read(sample_size)
            
            # 중간 부분
            f.seek(total_frames // 2)
            middle_data = f.read(sample_size)
            
            # 끝 부분
            f.seek(max(0, total_frames - sample_size))
            end_data = f.read(sample_size)
            
            # 샘플 데이터 합치기
            data = np.concatenate([start_data, middle_data, end_data])
            sr = f.samplerate
    except LibsndfileError:
        try:
            wav_buffer = BytesIO()
            audio_file = AudioSegment.from_file(deepcopy(buffer))
            audio_file.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            
            with sf.SoundFile(wav_buffer, 'r') as f:
                total_frames = f.frames
                sample_size = min(int(f.samplerate * 5), total_frames // 3)
                
                f.seek(0)
                start_data = f.read(sample_size)
                
                f.seek(total_frames // 2)
                middle_data = f.read(sample_size)
                
                f.seek(max(0, total_frames - sample_size))
                end_data = f.read(sample_size)
                
                data = np.concatenate([start_data, middle_data, end_data])
                sr = f.samplerate
        except Exception as e:
            return None

    # 다채널 오디오라면, 모노로 변환
    if data.ndim > 1:
        y = np.mean(data, axis=1)
    else:
        y = data

    # RMS 계산 (간소화)
    rms = np.sqrt(np.mean(y**2))
    
    # dB 스케일로 변환
    epsilon = 1e-12
    max_amp = np.max(np.abs(y))
    noise_floor_dbfs = 20 * np.log10(rms / max_amp + epsilon)

    return round(noise_floor_dbfs, 2)

# 스테레오 상태 확인 함수 (최적화 - 샘플링 사용)
def check_stereo_status_from_buffer(buffer):
    try:
        with sf.SoundFile(deepcopy(buffer), 'r') as f:
            # 채널 수만 확인하고, 모노인 경우 바로 반환
            if f.channels == 1:
                return "Mono"
            
            # 스테레오인 경우 일부만 샘플링하여 확인
            sample_size = min(int(f.samplerate * 2), f.frames)
            data = f.read(sample_size)
            
            if np.array_equal(data[:, 0], data[:, 1]):
                return "Dual Mono"
            else:
                return "True Stereo"
    except LibsndfileError:
        try:
            wav_buffer = BytesIO()
            audio_file = AudioSegment.from_file(deepcopy(buffer))
            audio_file.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            
            with sf.SoundFile(wav_buffer, 'r') as f:
                if f.channels == 1:
                    return "Mono"
                
                sample_size = min(int(f.samplerate * 2), f.frames)
                data = f.read(sample_size)
                
                if np.array_equal(data[:, 0], data[:, 1]):
                    return "Dual Mono"
                else:
                    return "True Stereo"
        except Exception as e:
            return f"Unknown (Error)"

# 메인 앱 코드
def main():
    st.title("Audio Requirements Validator")
    st.markdown("대량의 오디오 파일 사양을 검사하고 요구사항과 비교합니다.")
    
    # 세션 상태 초기화
    init_session_state()
    
    # 사이드바 설정
    st.sidebar.header("File requirements settings")
    required_format = st.sidebar.selectbox("Format", ["WAV", "MP3", "AAC"], disabled=st.session_state.disabled)
    required_channels = st.sidebar.selectbox("Channels", [1, 2], disabled=st.session_state.disabled)
    required_sample_rate = st.sidebar.selectbox("Sample Rate (Hz)", [44100, 48000, 96000, 192000], disabled=st.session_state.disabled)
    required_bit_depth = st.sidebar.selectbox("Bit Depth", [16, 24, 32], disabled=st.session_state.disabled)
    required_noise_floor = st.sidebar.slider("Noise Floor (dBFS)", min_value=-100, max_value=0, value=-60, disabled=st.session_state.disabled)
    required_stereo_status = st.sidebar.selectbox(
        "Stereo Status", ["Dual Mono", "Mono", "True Stereo"], disabled=st.session_state.disabled
    )
    
    # 버튼 설정
    sidebar_col1, sidebar_col2 = st.sidebar.columns(2)
    with sidebar_col1:
        required_start_button = st.sidebar.button("Save", key="start_button", use_container_width=True)
    with sidebar_col2:
        required_new_button = st.sidebar.button("Reset", key="new_button", use_container_width=True)
    
    # 버튼 동작 처리
    if required_start_button:
        st.session_state.required_format = required_format
        st.session_state.required_channels = required_channels
        st.session_state.required_sample_rate = required_sample_rate
        st.session_state.required_bit_depth = required_bit_depth
        st.session_state.required_noise_floor = required_noise_floor
        st.session_state.required_stereo_status = required_stereo_status
        st.session_state.start_button_clicked = True
        st.session_state.disabled = True
        st.rerun()
        
    if required_new_button:
        st.session_state.start_button_clicked = False
        st.session_state.disabled = False
        st.rerun()
    
    # 메인 처리 로직
    if st.session_state.start_button_clicked:
        st.header("1. 파일 업로드", divider="red")
        
        # 파일 업로드 위젯
        uploaded_files = st.file_uploader(
            "오디오 파일(wav, mp3, flac 등)을 업로드하세요. (최대 10개만 처리됩니다)",
            type=["wav", "mp3", "flac", "m4a", "aac", "ogg"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            # 최대 10개 파일만 처리
            MAX_FILES = 10
            if len(uploaded_files) > MAX_FILES:
                st.warning(f"업로드된 {len(uploaded_files)}개 파일 중 처음 {MAX_FILES}개만 처리됩니다.")
                uploaded_files = uploaded_files[:MAX_FILES]
            
            # 진행 상황 표시
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            # 파일 처리
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"처리 중: {idx+1}/{len(uploaded_files)} - {uploaded_file.name}")
                progress_bar.progress((idx) / len(uploaded_files))
                
                # 파일 버퍼 얻기
                audio_buffer = BytesIO(uploaded_file.getvalue())
                
                # 파일 유효성 검사
                is_valid_file, msg, file_type = validate_filetype(audio_buffer)
                if not is_valid_file:
                    results.append({
                        "Valid": "X",
                        "Name": uploaded_file.name,
                        "Time (sec)": "Error",
                        "Format": file_type,
                        "Sample Rate": "Error",
                        "Bit Depth": "Error",
                        "Channels": "Error",
                        "Stereo Status": "Error",
                        "Noise Floor (dBFS)": "Error",
                    })
                    continue
                
                # 오디오 속성 분석
                properties = get_audio_properties_from_buffer(deepcopy(audio_buffer))
                noise_floor_val = calculate_noise_floor_from_buffer(deepcopy(audio_buffer))
                stereo_status = check_stereo_status_from_buffer(deepcopy(audio_buffer))
                
                # 요구사항과 비교
                matches_format = file_type.lower() == st.session_state.required_format.lower()
                matches_channels = properties["Channels"] == st.session_state.required_channels if properties["Channels"] != "Error" else False
                matches_sample_rate = properties["Sample Rate"] == st.session_state.required_sample_rate if properties["Sample Rate"] != "Error" else False
                matches_bit_depth = str(properties["Bit Depth"]) == str(st.session_state.required_bit_depth) if properties["Bit Depth"] != "Error" else False
                matches_noise_floor = noise_floor_val < st.session_state.required_noise_floor if isinstance(noise_floor_val, (int, float)) else False
                matches_stereo_status = stereo_status == st.session_state.required_stereo_status if stereo_status != "Unknown (Error)" else False
                
                matches_all = all([
                    matches_format,
                    matches_channels,
                    matches_sample_rate,
                    matches_bit_depth,
                    matches_noise_floor,
                    matches_stereo_status,
                ])
                
                # 결과 저장
                results.append({
                    "Valid": "O" if matches_all else "X",
                    "Name": uploaded_file.name,
                    "Time (sec)": properties["Duration (seconds)"] if properties["Duration (seconds)"] != "Error (Processing Failed)" else "Error",
                    "Format": st.session_state.required_format if matches_format else f"{file_type}",
                    "Sample Rate": f"{properties['Sample Rate']} Hz" if properties["Sample Rate"] != "Error" else "Error",
                    "Bit Depth": properties["Bit Depth"] if properties["Bit Depth"] != "Error" else "Error",
                    "Channels": properties["Channels"] if properties["Channels"] != "Error" else "Error",
                    "Stereo Status": stereo_status,
                    "Noise Floor (dBFS)": noise_floor_val if isinstance(noise_floor_val, (int, float)) else "Error",
                })
                
                # 메모리 최적화: 처리 완료된 파일은 메모리에서 해제
                del audio_buffer
                gc.collect()
            
            # 진행 완료
            progress_bar.progress(1.0)
            status_text.text("처리 완료!")
            
            # 결과 표시
            st.header("2. 검사 결과", divider="blue")
            
            # 데이터프레임 생성 및 스타일링
            df_results = pd.DataFrame(results).reset_index(drop=True)
            
            # 셀 색상 지정 함수
            def highlight_rows(row):
                green = 'background-color: lightgreen;color: black;'
                red = 'background-color: lightcoral;color: black;'
                
                if row['Valid'] == "O":
                    colors = [green] * 6
                elif row["Time (sec)"] == "Error":
                    return [red] * len(row)
                else:
                    colors = [
                        green if row["Format"] == st.session_state.required_format else red,
                        green if row["Sample Rate"] == f'{st.session_state.required_sample_rate} Hz' else red,
                        green if row["Bit Depth"] == str(st.session_state.required_bit_depth) else red,
                        green if row["Channels"] == st.session_state.required_channels else red,
                        green if row["Stereo Status"] == st.session_state.required_stereo_status else red,
                        red if row["Noise Floor (dBFS)"] == "Error" or row["Noise Floor (dBFS)"] >= st.session_state.required_noise_floor else green
                    ]
                return [None] * (len(row) - 6) + colors
            
            # 스타일 적용 및 표시
            styled_df_results = df_results.style.apply(highlight_rows, axis=1).format(precision=2)
            st.table(styled_df_results)
            
            # 요약 정보 표시
            valid_count = df_results['Valid'].value_counts().get('O', 0)
            invalid_count = df_results['Valid'].value_counts().get('X', 0)
            
            st.subheader("요약")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("총 파일 수", len(df_results))
            with col2:
                st.metric("적합 파일 수", valid_count)
            with col3:
                st.metric("부적합 파일 수", invalid_count)
            
            # 메모리 최적화: 처리 완료 후 메모리 정리
            del results
            del df_results
            gc.collect()
        else:
            st.info("파일을 업로드하면 결과가 표시됩니다. (최대 10개 파일만 처리)")

if __name__ == "__main__":
    main()
