# VoiceSummary

LLM 기반 음성 요약 도구로, 오디오 파일에서 화자 분리(diarization)와 전사(transcription)를 수행합니다.

## 요구사항

- Python 3.13 이상
- [uv](https://github.com/astral-sh/uv) 패키지 관리자
- Hugging Face 토큰 (화자 분리 모델 접근용)
- CUDA 지원 GPU

## 설치

1. 저장소 클론:
```bash
git clone <repository-url>
cd VoiceSummary
```

2. 의존성 설치:
```bash
uv sync
```

## 환경 변수 설정

프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 Hugging Face 토큰을 추가하세요:

```env
HF_TOKEN=your_huggingface_token_here
```

Hugging Face 토큰은 [Hugging Face 설정 페이지](https://huggingface.co/settings/tokens)에서 발급받을 수 있습니다.

본 프로젝트는 [WhisperX](https://github.com/m-bain/whisperX)를 사용하므로 다음 모델들에 대한 구독이 필요합니다.

- [Segmentation](https://huggingface.co/pyannote/segmentation-3.0)
- [Speaker-Diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

## 사용 방법

### 기본 사용법

```bash
uv run python src/main.py --audio_path <오디오_파일_경로> --language <언어코드> --min_speakers <최소_화자_수> --max_speakers <최대_화자_수>
```

### 매개변수 설명

- `--audio_path` (필수): 처리할 오디오 파일의 경로
- `--language` (선택): 오디오 파일의 언어 코드 (기본값: `en`)
  - 예: `en` (영어), `ko` (한국어), `ja` (일본어), `fr` (프랑스어) 등
- `--min_speakers` (선택): 예상되는 최소 화자 수 (기본값: `1`)
- `--max_speakers` (선택): 예상되는 최대 화자 수 (기본값: `4`)

### 사용 예시

#### 영어 회의록 전사
```bash
uv run python src/main.py --audio_path test/test_meeting.mp3 --language en --min_speakers 2 --max_speakers 5
```

#### 한국어 인터뷰 전사
```bash
uv run python src/main.py --audio_path test/test_interview.mp3 --language ko --min_speakers 1 --max_speakers 2
```

## 출력 형식

프로그램은 각 세그먼트에 대해 다음과 같은 형식으로 결과를 출력합니다:

```
SPEAKER_00: 첫 번째 화자가 말한 내용입니다.
SPEAKER_01: 두 번째 화자가 말한 내용입니다.
SPEAKER_00: 다시 첫 번째 화자가 말한 내용입니다.
```

## 로그

프로그램 실행 시 로그는 다음 위치에 저장됩니다:
- 콘솔 출력
- `logs/voicesummary.log` 파일

## 문제 해결

### Hugging Face 인증 오류
- `.env` 파일에 올바른 `HF_TOKEN`이 설정되어 있는지 확인하세요.
- 토큰이 유효하고 화자 분리 모델 접근 권한이 있는지 확인하세요.

### GPU 메모리 부족
- `voice_module.py`의 `batch_size`를 줄이거나 `compute_type`을 `"int8"`로 변경하세요.

### 파일을 찾을 수 없음
- 오디오 파일 경로가 올바른지 확인하세요.
- 상대 경로 사용 시 현재 작업 디렉토리를 확인하세요.
