# 제품 요구사항 정의서 (PRD)
# Electrode 3D Generator

**버전**: 1.0.0
**최종 수정일**: 2026-01-21
**작성자**: AI Development Team

---

## 1. 개요

### 1.1 프로젝트 목적
2D 배터리 전극 SEM 이미지로부터 3D 미세구조를 자동 생성하고, 이를 COMSOL Multiphysics 시뮬레이션에 활용할 수 있는 통합 파이프라인을 구축한다.

### 1.2 핵심 가치 제안
- **시간 절감**: 기존 FIB-SEM 3D 스캐닝 대비 90% 이상 시간 단축
- **비용 절감**: 고가의 3D 이미징 장비 불필요
- **재현성**: 일관된 품질의 가상 미세구조 대량 생성 가능
- **통합 워크플로우**: 생성부터 시뮬레이션까지 자동화

### 1.3 대상 사용자
| 사용자 유형 | 설명 | 주요 니즈 |
|------------|------|----------|
| 배터리 연구원 | 대학/연구소 연구자 | 빠른 미세구조 분석 |
| 전극 개발자 | 기업 R&D 엔지니어 | 설계 최적화, 시뮬레이션 |
| 재료 과학자 | 신소재 개발자 | 구조-성능 상관관계 연구 |

---

## 2. 기능 요구사항

### 2.1 핵심 기능 (P0 - Must Have)

#### F-001: 2D 이미지 전처리
| 항목 | 설명 |
|------|------|
| **기능** | SEM 이미지 로딩, 노이즈 제거, 정규화, 상 분할 |
| **입력** | PNG, TIFF 형식 SEM 이미지 (그레이스케일) |
| **출력** | 분할된 다상 이미지 (0: 기공, 1: 활물질, 2: 바인더) |
| **처리량** | 단일 이미지 < 5초 |
| **정확도** | 상 분할 정확도 > 95% (수동 분할 대비) |

#### F-002: 3D 미세구조 생성 (SliceGAN)
| 항목 | 설명 |
|------|------|
| **기능** | 2D 이미지로부터 통계적으로 일관된 3D 복셀 볼륨 생성 |
| **입력** | 전처리된 2D 이미지 |
| **출력** | 64×64×64 복셀 볼륨 (확장 가능) |
| **훈련 시간** | < 2시간 (단일 GPU) |
| **생성 시간** | < 1초 per sample |
| **품질 지표** | 참조 데이터 대비 상 분율 오차 < 5% |

#### F-003: 복셀-메시 변환
| 항목 | 설명 |
|------|------|
| **기능** | Marching Cubes 알고리즘으로 복셀을 표면 메시로 변환 |
| **입력** | 3D 복셀 볼륨 |
| **출력** | 삼각형 메시 (STL, OBJ 형식) |
| **메시 품질** | Watertight, manifold 보장 |
| **정제 옵션** | 스무딩, 단순화, 리메싱 |

#### F-004: 미세구조 메트릭 계산
| 항목 | 설명 |
|------|------|
| **기능** | 전극 성능 관련 구조 특성 정량화 |
| **메트릭** | 기공률, 비표면적, 굴곡도, 연결성, 입자 크기 분포 |
| **출력 형식** | JSON, CSV |

#### F-005: 파일 내보내기
| 항목 | 설명 |
|------|------|
| **지원 형식** | STL (binary/ASCII), OBJ, VTK, NASTRAN |
| **좌표계 변환** | mm → m (COMSOL 호환) |
| **배치 처리** | 다중 메시 동시 내보내기 |

### 2.2 중요 기능 (P1 - Should Have)

#### F-006: COMSOL 연계
| 항목 | 설명 |
|------|------|
| **기능** | mph 라이브러리를 통한 COMSOL 자동화 |
| **지원 작업** | 메시 임포트, 재료 설정, 물리 설정, 시뮬레이션 실행 |
| **의존성** | COMSOL Multiphysics 6.0+ 설치 필요 |

#### F-007: TPMS 구조 생성
| 항목 | 설명 |
|------|------|
| **기능** | 수학적 정의 기반 TPMS 구조 생성 |
| **지원 유형** | Gyroid, Schwarz P, Schwarz D, I-WP, Neovius |
| **파라미터** | 주기, 두께, 해상도, 치수 |

#### F-008: 시각화
| 항목 | 설명 |
|------|------|
| **2D** | 슬라이스 뷰, 직교 뷰, 메트릭 차트 |
| **3D** | PyVista 기반 볼륨 렌더링 |
| **애니메이션** | GIF 내보내기 |

### 2.3 선택 기능 (P2 - Nice to Have)

#### F-009: 웹 인터페이스
- Gradio 기반 웹 UI
- 드래그 앤 드롭 이미지 업로드
- 실시간 3D 뷰어

#### F-010: 다중 해상도 생성
- 128×128×128, 256×256×256 볼륨 지원
- 메모리 효율적 생성

#### F-011: 전이 학습
- 사전 훈련된 모델 제공
- 적은 데이터로 fine-tuning

---

## 3. 비기능 요구사항

### 3.1 성능 요구사항
| 항목 | 요구사항 |
|------|----------|
| 이미지 전처리 | < 5초 / 이미지 |
| 3D 생성 (추론) | < 1초 / 볼륨 |
| 메시 변환 | < 10초 / 볼륨 |
| 메트릭 계산 | < 5초 / 볼륨 |
| 메모리 사용 | < 8GB GPU VRAM (훈련) |

### 3.2 호환성 요구사항
| 항목 | 요구사항 |
|------|----------|
| Python | 3.9 - 3.11 |
| PyTorch | 2.0+ |
| OS | Windows 10/11, Linux (Ubuntu 20.04+) |
| GPU | CUDA 11.8+ (선택, CPU 지원) |
| COMSOL | 6.0+ (선택) |

### 3.3 안정성 요구사항
- 테스트 커버리지 > 80%
- CI/CD 파이프라인 구축
- 에러 로깅 및 복구 메커니즘

### 3.4 보안 요구사항
- 로컬 실행 (데이터 외부 전송 없음)
- 민감 정보 환경 변수 관리
- 의존성 취약점 정기 검사

---

## 4. 기술 아키텍처

### 4.1 시스템 구성도

```
┌─────────────────────────────────────────────────────────────┐
│                    Electrode 3D Generator                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  CLI/API    │    │  Web UI      │    │  Jupyter     │   │
│  │  Interface  │    │  (Gradio)    │    │  Notebooks   │   │
│  └──────┬──────┘    └──────┬───────┘    └──────┬───────┘   │
│         │                  │                   │            │
│         └──────────────────┼───────────────────┘            │
│                            │                                │
│  ┌─────────────────────────┴─────────────────────────────┐  │
│  │                   Core Engine                          │  │
│  ├───────────────────────────────────────────────────────┤  │
│  │                                                       │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │  │
│  │  │Preprocessing│  │  SliceGAN   │  │Postprocessing│   │  │
│  │  │   Module    │  │   Model     │  │   Module    │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘   │  │
│  │                                                       │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │  │
│  │  │   Utils     │  │   COMSOL    │  │Visualization│   │  │
│  │  │  (Metrics)  │  │  Interface  │  │             │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘   │  │
│  │                                                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                External Dependencies                   │  │
│  │  PyTorch | trimesh | scikit-image | mph | PyVista     │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 데이터 흐름

```
2D SEM Image
     │
     ▼
┌──────────────┐
│ Load Image   │ ← PNG, TIFF
│ (PIL/tifffile)│
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Denoise      │ ← Gaussian filter
│ Normalize    │
│ Segment      │ ← Threshold/K-means/Otsu
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ SliceGAN     │ ← WGAN-GP Training
│ Training     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 3D Volume    │ ← Generator inference
│ Generation   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Marching     │ ← scikit-image
│ Cubes        │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Mesh         │ ← trimesh
│ Refinement   │ ← Smooth, Repair
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Export       │ → STL, OBJ, VTK
│              │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ COMSOL       │ → Simulation
│ Integration  │
└──────────────┘
```

### 4.3 모듈 구조

```
electrode_generator/
├── __init__.py
├── config.py          # 설정 클래스
├── core.py            # 메인 파이프라인
└── cli.py             # CLI 인터페이스

models/
├── slicegan/
│   ├── generator.py   # 3D Generator
│   ├── discriminator.py  # 2D Critic
│   └── trainer.py     # 훈련 로직

preprocessing/
├── image_processor.py # 이미지 전처리
└── augmentation.py    # 데이터 증강

postprocessing/
├── mesh_converter.py  # Marching Cubes
└── export.py          # 파일 내보내기

comsol/
├── interface.py       # COMSOL 연결
└── simulation.py      # 시뮬레이션 설정

utils/
├── metrics.py         # 메트릭 계산
└── visualization.py   # 시각화
```

---

## 5. 데이터 요구사항

### 5.1 입력 데이터 사양
| 항목 | 사양 |
|------|------|
| 형식 | PNG, TIFF (8-bit, 16-bit) |
| 해상도 | 최소 256×256, 권장 512×512+ |
| 색상 | 그레이스케일 (RGB 자동 변환) |
| 상 수 | 2-4개 (기공, 활물질, 바인더, 첨가제) |

### 5.2 출력 데이터 사양
| 항목 | 사양 |
|------|------|
| 3D 볼륨 | NumPy 배열 (uint8), 64³-256³ |
| 메시 | STL (binary), OBJ, VTK |
| 메트릭 | JSON, CSV |
| 이미지 | PNG (슬라이스), GIF (애니메이션) |

### 5.3 데이터 저장 구조
```
data/
├── raw/           # 원본 SEM 이미지
├── processed/     # 전처리된 이미지
└── generated/     # 생성된 3D 구조
    ├── volumes/   # .npy 파일
    ├── meshes/    # .stl, .obj 파일
    └── metrics/   # .json 파일
```

---

## 6. 사용자 인터페이스

### 6.1 CLI 명령어

```bash
# 전처리
electrode-gen preprocess input.png -o processed.png --segment

# 훈련
electrode-gen train --input processed.png --epochs 100 --output model.pt

# 생성
electrode-gen generate --model model.pt --num-samples 10 --output ./volumes/

# 메시 변환
electrode-gen mesh volume.npy --smooth --output electrode.stl

# 메트릭 계산
electrode-gen metrics volume.npy --output metrics.json

# 전체 파이프라인
electrode-gen pipeline input.png --output-dir ./results/
```

### 6.2 Python API

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

---

## 7. 테스트 요구사항

### 7.1 단위 테스트
| 모듈 | 테스트 항목 |
|------|------------|
| config | YAML 로드/저장, 기본값 |
| preprocessing | 이미지 로드, 정규화, 분할 |
| slicegan | Generator/Critic 출력 형상, 손실 계산 |
| postprocessing | Marching Cubes, 메시 내보내기 |
| metrics | 기공률, 비표면적, 굴곡도 |

### 7.2 통합 테스트
- 전체 파이프라인 end-to-end 실행
- 다양한 입력 이미지 형식 처리
- 대용량 볼륨 생성 메모리 테스트

### 7.3 성능 테스트
- 훈련 시간 벤치마크
- 추론 속도 벤치마크
- GPU 메모리 사용량

---

## 8. 릴리스 계획

### 8.1 MVP (v0.1.0)
- [x] 기본 전처리 파이프라인
- [x] SliceGAN 훈련/추론
- [x] Marching Cubes 메시 변환
- [x] STL/OBJ 내보내기
- [x] 기본 메트릭 계산

### 8.2 v0.2.0
- [ ] COMSOL 연계 완성
- [ ] 웹 UI (Gradio)
- [ ] TPMS 구조 생성
- [ ] 사전 훈련된 모델

### 8.3 v1.0.0
- [ ] 다중 해상도 지원
- [ ] 전이 학습
- [ ] 문서화 완성
- [ ] PyPI 패키지 배포

---

## 9. 참고 자료

### 9.1 논문
1. Kench, S., & Cooper, S. J. (2021). Generating three-dimensional structures from a two-dimensional slice with generative adversarial network-based dimensionality expansion. *Nature Machine Intelligence*, 3(4), 299-305.

2. Müller, S., et al. (2022). MicroLib: A library of 3D microstructures generated from 2D micrographs using SliceGAN. *Scientific Data*, 9, 645.

### 9.2 오픈소스
- SliceGAN: https://github.com/stke9/SliceGAN
- trimesh: https://github.com/mikedh/trimesh
- mph: https://github.com/MPh-py/MPh

### 9.3 데이터셋
- NREL Battery Microstructure Library

---

## 10. 용어 정의

| 용어 | 정의 |
|------|------|
| SEM | Scanning Electron Microscopy, 주사전자현미경 |
| FIB-SEM | Focused Ion Beam SEM, 집속이온빔 주사전자현미경 |
| WGAN-GP | Wasserstein GAN with Gradient Penalty |
| TPMS | Triply Periodic Minimal Surfaces |
| Marching Cubes | 복셀 데이터를 등위면 메시로 변환하는 알고리즘 |
| Tortuosity | 굴곡도, 실제 경로 길이 / 직선 거리 |
| Porosity | 기공률, 기공 부피 / 전체 부피 |

---

*문서 끝*
