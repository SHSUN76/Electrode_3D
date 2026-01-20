# 전극 구조 자동생성기 (Electrode Structure Auto-Generator)

## 프로젝트 개요

이 프로젝트는 배터리 전극의 3D 미세구조를 자동으로 생성하고, COMSOL Multiphysics 시뮬레이션과 연계하는 시스템입니다.

**GitHub Repository**: https://github.com/SHSUN76/Electrode_3D

## 핵심 문서

| 문서 | 설명 | 경로 |
|------|------|------|
| **PRD** | 제품 요구사항 정의서 | [docs/PRD.md](docs/PRD.md) |
| **ERD** | 엔티티 관계 다이어그램 | [docs/ERD.md](docs/ERD.md) |
| **MVP** | 최소 기능 제품 정의 | [docs/MVP.md](docs/MVP.md) |
| **기술 참고자료** | 기술 스택 상세 문서 | [참고자료1.md](참고자료1.md) |

## 기술 스택

- **3D 구조 생성**: SliceGAN (PyTorch), TPMS 기반 구조 생성
- **이미지 분할**: 3D U-Net, Swin UNETR, K-means, Multi-Otsu
- **메시 처리**: trimesh, PyVista, Blender Python API (bpy)
- **FEM 메시**: Gmsh Python API
- **시뮬레이션**: COMSOL Multiphysics (mph 라이브러리)

## 주요 파일 구조

```
전극_구조_자동생성기/
├── CLAUDE.md                    # 프로젝트 설명 (이 파일)
├── .claude/                     # Claude Code 설정
├── pyproject.toml               # Python 프로젝트 설정
├── requirements.txt             # 의존성 목록
├── 참고자료1.md                 # 기술 스택 상세 문서
│
├── docs/                        # 문서
│   ├── PRD.md                  # 제품 요구사항 정의서
│   └── ERD.md                  # 엔티티 관계 다이어그램
│
├── electrode_generator/         # 핵심 패키지
│   ├── __init__.py
│   ├── config.py               # 설정 클래스
│   ├── core.py                 # 메인 파이프라인
│   └── cli.py                  # CLI 인터페이스
│
├── models/                      # 모델 구현
│   └── slicegan/
│       ├── generator.py        # 3D Generator
│       ├── discriminator.py    # 2D Critic (WGAN-GP)
│       └── trainer.py          # 학습 로직
│
├── preprocessing/               # 이미지 전처리
│   ├── image_processor.py      # 이미지 로드/정규화/분할
│   └── augmentation.py         # 데이터 증강
│
├── postprocessing/              # 후처리
│   ├── mesh_converter.py       # Marching Cubes, TPMS
│   └── export.py               # STL/OBJ/VTK 내보내기
│
├── comsol/                      # COMSOL 연계
│   ├── interface.py            # mph 라이브러리 인터페이스
│   └── simulation.py           # 전기화학 시뮬레이션
│
├── utils/                       # 유틸리티
│   ├── metrics.py              # 미세구조 메트릭 계산
│   └── visualization.py        # 시각화
│
├── tests/                       # 테스트
│   ├── test_config.py
│   ├── test_slicegan.py
│   ├── test_preprocessing.py
│   ├── test_postprocessing.py
│   └── test_metrics.py
│
├── data/                        # 데이터 디렉토리
│   ├── raw/                    # 원본 SEM 이미지
│   ├── processed/              # 전처리된 데이터
│   └── generated/              # 생성된 3D 구조
│
└── experiments/                 # 실험 결과
```

## 빠른 시작

### 설치

```bash
# 저장소 클론
git clone https://github.com/SHSUN76/Electrode_3D.git
cd Electrode_3D

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -e .

# 또는 개발 의존성 포함
pip install -e ".[dev]"
```

### CLI 사용법

```bash
# 전처리
electrode-gen preprocess input.png -o processed.png --segment

# 모델 훈련
electrode-gen train --input processed.png --epochs 100 --output model.pt

# 3D 구조 생성
electrode-gen generate --model model.pt --num-samples 10 --output ./volumes/

# 메시 변환
electrode-gen mesh volume.npy --smooth --output electrode.stl

# 메트릭 계산
electrode-gen metrics volume.npy --output metrics.json

# 전체 파이프라인
electrode-gen pipeline input.png --output-dir ./results/
```

### Python API 사용법

```python
from electrode_generator import ElectrodeGenerator, Config

# 설정 로드
config = Config.load_yaml("config.yaml")

# 생성기 초기화
generator = ElectrodeGenerator(config)

# 파이프라인 실행
results = generator.run_pipeline(
    input_path="sem_image.png",
    output_dir="./output",
)

# 결과 확인
print(f"생성된 볼륨: {results['volume'].shape}")
print(f"메시 경로: {results['mesh_path']}")
print(f"기공률: {results['metrics']['porosity']:.2%}")
```

## 개발 지침

### 언어
- 주 언어: Python 3.10+
- 문서: 한국어 (기술 용어는 영문 병기)

### 코드 스타일
- PEP 8 준수
- Type hints 사용 필수
- Docstring: Google 스타일
- 테스트 커버리지 > 80% 목표

### 테스트 실행

```bash
# 전체 테스트
pytest

# 커버리지 포함
pytest --cov=electrode_generator --cov-report=html

# 특정 모듈 테스트
pytest tests/test_slicegan.py -v
```

## 워크플로우

```
2D SEM 이미지 → 이미지 분할 → SliceGAN 학습 → 3D 복셀 생성
                                                    ↓
COMSOL 시뮬레이션 ← FEM 메시 생성 ← 메시 정제 ← Marching Cubes
```

## 핵심 라이브러리

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| PyTorch | >=2.0 | SliceGAN 모델 |
| NumPy | - | 배열 처리 |
| scikit-image | - | Marching Cubes, 이미지 처리 |
| trimesh | - | 메시 처리 |
| PyVista | - | 3D 시각화 |
| mph | - | COMSOL 연계 (선택) |

## 참고 자료

- [PRD - 제품 요구사항 정의서](docs/PRD.md)
- [ERD - 엔티티 관계 다이어그램](docs/ERD.md)
- [기술 참고자료](참고자료1.md): SliceGAN, Blender, COMSOL 상세
- [SliceGAN 논문](https://www.nature.com/articles/s42256-021-00322-1)
- [SliceGAN GitHub](https://github.com/stke9/SliceGAN)
- [COMSOL Battery Design Module](https://www.comsol.com/battery-design-module)

## 주의사항

- COMSOL 라이선스 필요 (mph 라이브러리 사용 시)
- GPU 권장 (SliceGAN 학습용, CUDA 11.8+)
- Blender 4.x 버전 사용 권장 (bpy 사용 시)

## 라이선스

MIT License
