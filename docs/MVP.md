# MVP (Minimum Viable Product) 정의
# Electrode 3D Generator

**버전**: 1.0.0
**최종 수정일**: 2026-01-21

---

## 1. MVP 개요

### 1.1 MVP 목표
2D 배터리 전극 SEM 이미지 하나로부터 3D 미세구조를 생성하고,
메시로 변환하여 COMSOL 시뮬레이션에 사용할 수 있는 최소 기능 제품을 구현합니다.

### 1.2 MVP 범위

#### 포함 (In Scope)
| 기능 | 설명 | 우선순위 |
|------|------|----------|
| 이미지 전처리 | 로드, 노이즈 제거, 정규화, 상 분할 | P0 |
| SliceGAN 훈련 | WGAN-GP 기반 3D 생성 모델 훈련 | P0 |
| 3D 볼륨 생성 | 64×64×64 복셀 볼륨 생성 | P0 |
| Marching Cubes | 복셀 → 표면 메시 변환 | P0 |
| 메시 내보내기 | STL, OBJ 형식 지원 | P0 |
| 기본 메트릭 | 기공률, 비표면적 계산 | P0 |
| CLI | 기본 명령줄 인터페이스 | P0 |

#### 제외 (Out of Scope - v1.0 이후)
| 기능 | 이유 |
|------|------|
| 웹 UI | 추가 개발 시간 필요 |
| 다중 해상도 (128³, 256³) | 메모리/시간 최적화 필요 |
| 자동 COMSOL 시뮬레이션 | COMSOL 라이선스 의존 |
| 사전 훈련된 모델 | 공개 데이터셋으로 추가 훈련 필요 |
| 전이 학습 | 추가 연구 필요 |

### 1.3 성공 기준

| 기준 | 목표값 | 측정 방법 |
|------|--------|----------|
| 훈련 완료 | 100 에폭 이내 | 손실 수렴 확인 |
| 생성 시간 | < 1초/볼륨 | 벤치마크 |
| 메시 품질 | Watertight | trimesh 검증 |
| 기공률 정확도 | ±5% | 참조 대비 오차 |
| 테스트 통과 | 100% | pytest 실행 |

---

## 2. MVP 아키텍처

### 2.1 시스템 다이어그램

```
┌─────────────────────────────────────────────────────────────┐
│                      MVP Architecture                        │
└─────────────────────────────────────────────────────────────┘

                         ┌──────────────┐
                         │   CLI Input  │
                         │   (click)    │
                         └──────┬───────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────┐
│                    ElectrodeGenerator                      │
│                        (core.py)                           │
├───────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │  Preprocess │───▶│   Train     │───▶│  Generate   │   │
│  │   Module    │    │  SliceGAN   │    │  3D Volume  │   │
│  └─────────────┘    └─────────────┘    └──────┬──────┘   │
│                                               │          │
│  ┌─────────────┐    ┌─────────────┐    ┌──────┴──────┐   │
│  │   Export    │◀───│  Mesh       │◀───│  Marching   │   │
│  │   STL/OBJ   │    │  Refinement │    │   Cubes     │   │
│  └─────────────┘    └─────────────┘    └─────────────┘   │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │                    Metrics                           │ │
│  │    Porosity | SSA | Tortuosity | Connectivity       │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                           │
└───────────────────────────────────────────────────────────┘
                                │
                                ▼
                         ┌──────────────┐
                         │   Output     │
                         │  .stl/.obj   │
                         │  .json       │
                         └──────────────┘
```

### 2.2 데이터 흐름

```
Input: 2D SEM Image (PNG/TIFF, 256×256+)
                │
                ▼
┌───────────────────────────────┐
│ 1. Preprocessing              │
│    - Load image               │
│    - Gaussian denoise         │
│    - Normalize [0,1]          │
│    - Segment phases (3-class) │
│    Output: (3, 256, 256) tensor│
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ 2. SliceGAN Training          │
│    - Extract 64×64 patches    │
│    - Train Generator (5 layers)│
│    - Train Critic (WGAN-GP)   │
│    - 100 epochs, ~1-2 hours   │
│    Output: model checkpoint   │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ 3. 3D Generation              │
│    - Sample z ~ N(0,1)        │
│    - Generator forward pass   │
│    - Softmax → argmax         │
│    Output: (64, 64, 64) uint8 │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ 4. Marching Cubes             │
│    - Extract isosurface       │
│    - Per-phase extraction     │
│    Output: vertices, faces    │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ 5. Mesh Refinement            │
│    - Laplacian smoothing      │
│    - Hole filling             │
│    - Watertight repair        │
│    Output: trimesh object     │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ 6. Export                     │
│    - Save STL (binary)        │
│    - Save OBJ (optional)      │
│    Output: electrode.stl      │
└───────────────────────────────┘
```

---

## 3. MVP 기능 상세

### 3.1 이미지 전처리 모듈

**파일**: `preprocessing/image_processor.py`

```python
class ImagePreprocessor:
    """MVP 전처리 파이프라인"""

    def load(self, path: str) -> np.ndarray:
        """PNG/TIFF 이미지 로드"""

    def denoise(self, image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Gaussian 노이즈 제거"""

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """[0, 1] 정규화"""

    def segment_phases(self, image: np.ndarray, num_classes: int = 3) -> np.ndarray:
        """Multi-threshold 상 분할"""

    def load_and_preprocess(self, path: str) -> np.ndarray:
        """전체 파이프라인"""
```

**입력/출력 사양**:
| 항목 | 사양 |
|------|------|
| 입력 형식 | PNG, TIFF (8-bit grayscale) |
| 입력 크기 | 최소 256×256, 권장 512×512 |
| 출력 형상 | (num_classes, H, W) one-hot |
| 처리 시간 | < 5초 |

### 3.2 SliceGAN 모델

**파일**: `models/slicegan/`

#### Generator (3D)
```python
class Generator3D(nn.Module):
    """
    3D Transpose Convolution Generator

    Input: (batch, nz, 4, 4, 4) latent vectors
    Output: (batch, nc, 64, 64, 64) voxel volume

    Architecture:
        nz×4×4×4 → 512×8×8×8 → 256×16×16×16 → 128×32×32×32 → 64×64×64×64 → nc×64×64×64
    """
```

#### Critic (2D)
```python
class Critic2D(nn.Module):
    """
    2D Convolution Critic for WGAN-GP

    Input: (batch, nc, 64, 64) 2D slices
    Output: (batch, 1) critic score

    Architecture:
        nc×64×64 → 64×32×32 → 128×16×16 → 256×8×8 → 512×4×4 → 1
    """
```

#### Trainer
```python
class SliceGAN:
    """
    SliceGAN 훈련 및 추론 클래스

    Methods:
        train(image, epochs): 모델 훈련
        generate(num_samples): 3D 볼륨 생성
        save(path): 체크포인트 저장
        load(path): 체크포인트 로드
    """
```

**모델 사양**:
| 하이퍼파라미터 | 기본값 | 설명 |
|---------------|--------|------|
| nz | 64 | 잠재 벡터 차원 |
| ngf | 64 | Generator 기본 필터 수 |
| ndf | 64 | Critic 기본 필터 수 |
| nc | 3 | 출력 채널 (상 수) |
| lr_g | 0.0001 | Generator 학습률 |
| lr_d | 0.0004 | Critic 학습률 |
| lambda_gp | 10 | Gradient penalty 가중치 |
| n_critic | 5 | Critic 업데이트 비율 |

### 3.3 Marching Cubes 변환

**파일**: `postprocessing/mesh_converter.py`

```python
class VoxelToMesh:
    """
    Marching Cubes 복셀 → 메시 변환

    Methods:
        convert(voxels, phase_id): 단일 상 변환
        convert_multiphase(voxels): 모든 상 변환
        to_trimesh(voxels, phase_id): trimesh 객체 반환
    """
```

**변환 사양**:
| 항목 | 사양 |
|------|------|
| 입력 | (D, H, W) uint8 array |
| 출력 | vertices (N, 3), faces (M, 3) |
| 알고리즘 | scikit-image marching_cubes |
| 스무딩 | Laplacian (선택) |

### 3.4 메시 내보내기

**파일**: `postprocessing/export.py`

```python
class MeshExporter:
    """
    메시 파일 내보내기

    Methods:
        export_stl_binary(vertices, faces, path): 바이너리 STL
        export_stl_ascii(vertices, faces, path): ASCII STL
        export_obj(vertices, faces, path): OBJ 형식
    """
```

**지원 형식**:
| 형식 | 용도 | 크기 |
|------|------|------|
| STL (binary) | COMSOL, 일반 | 작음 |
| STL (ASCII) | 디버깅 | 큼 |
| OBJ | Blender, 일반 | 중간 |

### 3.5 메트릭 계산

**파일**: `utils/metrics.py`

```python
class MicrostructureMetrics:
    """
    미세구조 메트릭 계산

    Methods:
        porosity(voxels): 기공률
        specific_surface_area(voxels): 비표면적
        tortuosity_geometric(voxels): 굴곡도 (근사)
        connectivity(voxels): 연결성 분석
    """
```

**MVP 메트릭**:
| 메트릭 | 설명 | 단위 |
|--------|------|------|
| Porosity | 기공 부피 분율 | - |
| SSA | 비표면적 | 1/μm |
| Tortuosity | 경로 굴곡도 | - |
| Connectivity | 최대 연결 비율 | - |

### 3.6 CLI 인터페이스

**파일**: `electrode_generator/cli.py`

```bash
# MVP CLI 명령어

# 전체 파이프라인
electrode-gen pipeline input.png --output-dir ./results/

# 개별 단계
electrode-gen preprocess input.png -o processed.npy
electrode-gen train --input processed.npy --epochs 100 --output model.pt
electrode-gen generate --model model.pt --num-samples 5 --output ./volumes/
electrode-gen mesh volume.npy --output electrode.stl
electrode-gen metrics volume.npy --output metrics.json
```

---

## 4. MVP 테스트 계획

### 4.1 단위 테스트

| 모듈 | 테스트 | 파일 |
|------|--------|------|
| config | YAML 로드/저장 | test_config.py |
| preprocessing | 이미지 로드, 정규화 | test_preprocessing.py |
| slicegan | Generator/Critic 출력 형상 | test_slicegan.py |
| postprocessing | Marching Cubes, Export | test_postprocessing.py |
| metrics | 기공률, SSA 계산 | test_metrics.py |

### 4.2 통합 테스트

```python
def test_full_pipeline():
    """전체 파이프라인 통합 테스트"""
    # 1. 샘플 이미지 생성
    test_image = create_synthetic_electrode_image(256, 256)

    # 2. 전처리
    processor = ImagePreprocessor()
    processed = processor.load_and_preprocess(test_image)
    assert processed.shape == (3, 256, 256)

    # 3. 모델 훈련 (짧은 에폭)
    model = SliceGAN()
    model.train(processed, epochs=10)

    # 4. 볼륨 생성
    volume = model.generate(num_samples=1)[0]
    assert volume.shape == (64, 64, 64)

    # 5. 메시 변환
    converter = VoxelToMesh()
    verts, faces = converter.convert(volume, phase_id=1)
    assert len(verts) > 0
    assert len(faces) > 0

    # 6. 메트릭 계산
    metrics = MicrostructureMetrics()
    porosity = metrics.porosity(volume)
    assert 0 <= porosity <= 1
```

### 4.3 성능 벤치마크

```python
def benchmark_generation():
    """생성 속도 벤치마크"""
    model = SliceGAN()
    model.load("trained_model.pt")

    times = []
    for _ in range(100):
        start = time.time()
        model.generate(num_samples=1)
        times.append(time.time() - start)

    avg_time = np.mean(times)
    assert avg_time < 1.0, f"Generation too slow: {avg_time:.2f}s"
```

---

## 5. MVP 개발 일정

### 5.1 마일스톤

| 주차 | 목표 | 산출물 |
|------|------|--------|
| 1 | 프로젝트 구조, 설정 | pyproject.toml, 폴더 구조 |
| 2 | 전처리 모듈 | preprocessing/ 완성 |
| 3 | SliceGAN 구현 | models/slicegan/ 완성 |
| 4 | 후처리, 내보내기 | postprocessing/ 완성 |
| 5 | 메트릭, CLI | utils/, cli.py 완성 |
| 6 | 통합 테스트, 문서화 | tests/, docs/ 완성 |

### 5.2 현재 진행 상황

```
[✓] Phase 1: 프로젝트 기반 구축
[✓] Phase 2: PRD, ERD 문서 작성
[ ] Phase 3: MCP 서버 설정
[→] Phase 4: MVP 모델 정의 (이 문서)
[✓] Phase 5: 핵심 모듈 구현
[✓] Phase 6: COMSOL 연계 모듈
[ ] Phase 7: 테스트 및 검증
[ ] Phase 8: 문서화 및 최종 정리
```

---

## 6. MVP 사용 시나리오

### 6.1 기본 워크플로우

```bash
# 1. 환경 설정
cd Electrode_3D
pip install -e .

# 2. 샘플 데이터로 전체 파이프라인 실행
electrode-gen pipeline data/raw/sample_sem.png \
    --output-dir ./results \
    --epochs 50

# 3. 결과 확인
ls ./results/
# model.pt           - 학습된 모델
# volume_0.npy       - 생성된 3D 볼륨
# electrode.stl      - 메시 파일
# metrics.json       - 계산된 메트릭

# 4. 추가 샘플 생성
electrode-gen generate \
    --model ./results/model.pt \
    --num-samples 10 \
    --output ./results/volumes/
```

### 6.2 Python API 워크플로우

```python
from electrode_generator import ElectrodeGenerator, Config

# 설정
config = Config()
config.slicegan.epochs = 100
config.slicegan.batch_size = 8

# 생성기 초기화
gen = ElectrodeGenerator(config)

# 이미지 로드 및 전처리
gen.load_image("sem_image.png")

# 모델 훈련
gen.train()

# 3D 구조 생성
volume = gen.generate()

# 메시 변환 및 내보내기
gen.voxel_to_mesh(volume)
gen.export_mesh("output/electrode.stl")

# 메트릭 계산
metrics = gen.calculate_metrics(volume)
print(f"Porosity: {metrics['porosity']:.2%}")
print(f"SSA: {metrics['specific_surface_area']:.2f} 1/μm")
```

---

## 7. MVP 이후 로드맵

### v0.2.0 (Post-MVP)
- [ ] COMSOL 자동 연계 완성
- [ ] 웹 UI (Gradio)
- [ ] TPMS 구조 생성
- [ ] 사전 훈련 모델 제공

### v1.0.0
- [ ] 다중 해상도 지원 (128³, 256³)
- [ ] 전이 학습
- [ ] PyPI 패키지 배포
- [ ] Docker 이미지

---

## 8. 리스크 및 대응

| 리스크 | 영향 | 대응 방안 |
|--------|------|----------|
| GPU 메모리 부족 | 훈련 실패 | 배치 크기 감소, 체크포인트 |
| 수렴 실패 | 품질 저하 | 하이퍼파라미터 튜닝 |
| 메시 비수밀 | COMSOL 임포트 실패 | pymeshfix 후처리 |
| 의존성 충돌 | 설치 실패 | 가상환경, Docker |

---

*문서 끝*
