# 엔티티 관계 다이어그램 (ERD)
# Electrode 3D Generator

**버전**: 1.0.0
**최종 수정일**: 2026-01-21

---

## 1. 개요

본 문서는 Electrode 3D Generator 시스템의 데이터 모델과 엔티티 간의 관계를 정의합니다.

---

## 2. 엔티티 관계 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ELECTRODE 3D GENERATOR ERD                       │
└─────────────────────────────────────────────────────────────────────────┘

┌───────────────┐       1:N       ┌───────────────┐       1:N       ┌───────────────┐
│   Project     │────────────────▶│    Image      │────────────────▶│ ProcessedData │
│               │                 │               │                 │               │
│ - id (PK)     │                 │ - id (PK)     │                 │ - id (PK)     │
│ - name        │                 │ - project_id  │                 │ - image_id    │
│ - created_at  │                 │   (FK)        │                 │   (FK)        │
│ - config_path │                 │ - path        │                 │ - segmented   │
│               │                 │ - format      │                 │ - num_phases  │
└───────────────┘                 │ - resolution  │                 │ - metadata    │
                                  │ - created_at  │                 └───────┬───────┘
                                  └───────────────┘                         │
                                                                            │
                                                                            │ 1:N
                                                                            ▼
┌───────────────┐       N:1       ┌───────────────┐       1:1       ┌───────────────┐
│  ModelConfig  │◀───────────────│    Model      │────────────────▶│ TrainingLog   │
│               │                 │               │                 │               │
│ - id (PK)     │                 │ - id (PK)     │                 │ - id (PK)     │
│ - nz          │                 │ - config_id   │                 │ - model_id    │
│ - ngf         │                 │   (FK)        │                 │   (FK)        │
│ - ndf         │                 │ - checkpoint  │                 │ - epoch       │
│ - nc          │                 │ - trained_at  │                 │ - g_loss      │
│ - epochs      │                 │ - best_loss   │                 │ - d_loss      │
│ - lr_g        │                 └───────┬───────┘                 │ - timestamp   │
│ - lr_d        │                         │                         └───────────────┘
│ - lambda_gp   │                         │
└───────────────┘                         │ 1:N
                                          ▼
                                  ┌───────────────┐
                                  │   Volume      │
                                  │               │
                                  │ - id (PK)     │
                                  │ - model_id    │
                                  │   (FK)        │
                                  │ - shape       │
                                  │ - dtype       │
                                  │ - path        │
                                  │ - created_at  │
                                  └───────┬───────┘
                                          │
                          ┌───────────────┼───────────────┐
                          │               │               │
                          │ 1:N           │ 1:1           │ 1:N
                          ▼               ▼               ▼
                  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
                  │    Mesh       │ │   Metrics     │ │ Visualization │
                  │               │ │               │ │               │
                  │ - id (PK)     │ │ - id (PK)     │ │ - id (PK)     │
                  │ - volume_id   │ │ - volume_id   │ │ - volume_id   │
                  │   (FK)        │ │   (FK)        │ │   (FK)        │
                  │ - format      │ │ - porosity    │ │ - type        │
                  │ - vertices    │ │ - ssa         │ │ - path        │
                  │ - faces       │ │ - tortuosity  │ │ - settings    │
                  │ - is_watertight│ │ - connectivity│ │               │
                  │ - path        │ │ - psd         │ │               │
                  └───────┬───────┘ └───────────────┘ └───────────────┘
                          │
                          │ 1:N
                          ▼
                  ┌───────────────┐
                  │  Simulation   │
                  │               │
                  │ - id (PK)     │
                  │ - mesh_id     │
                  │   (FK)        │
                  │ - tool        │
                  │ - parameters  │
                  │ - results_path│
                  │ - status      │
                  │ - created_at  │
                  └───────────────┘
```

---

## 3. 엔티티 상세 정의

### 3.1 Project (프로젝트)

프로젝트는 하나의 연구/실험 단위를 나타냅니다.

| 필드 | 타입 | 제약조건 | 설명 |
|------|------|----------|------|
| id | UUID | PK | 고유 식별자 |
| name | VARCHAR(255) | NOT NULL | 프로젝트명 |
| description | TEXT | NULLABLE | 프로젝트 설명 |
| created_at | TIMESTAMP | NOT NULL | 생성 일시 |
| updated_at | TIMESTAMP | NOT NULL | 수정 일시 |
| config_path | VARCHAR(512) | NULLABLE | 설정 파일 경로 |

```python
@dataclass
class Project:
    id: str
    name: str
    description: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    config_path: Optional[str] = None
```

### 3.2 Image (원본 이미지)

SEM 원본 이미지 정보를 저장합니다.

| 필드 | 타입 | 제약조건 | 설명 |
|------|------|----------|------|
| id | UUID | PK | 고유 식별자 |
| project_id | UUID | FK → Project | 소속 프로젝트 |
| path | VARCHAR(512) | NOT NULL | 파일 경로 |
| format | VARCHAR(10) | NOT NULL | 파일 형식 (png, tiff) |
| width | INTEGER | NOT NULL | 이미지 너비 (픽셀) |
| height | INTEGER | NOT NULL | 이미지 높이 (픽셀) |
| bit_depth | INTEGER | NOT NULL | 비트 깊이 (8, 16) |
| pixel_size | FLOAT | NULLABLE | 픽셀 크기 (μm) |
| created_at | TIMESTAMP | NOT NULL | 업로드 일시 |
| metadata | JSON | NULLABLE | 추가 메타데이터 |

```python
@dataclass
class Image:
    id: str
    project_id: str
    path: str
    format: str
    width: int
    height: int
    bit_depth: int = 8
    pixel_size: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None
```

### 3.3 ProcessedData (전처리 데이터)

전처리된 이미지 정보를 저장합니다.

| 필드 | 타입 | 제약조건 | 설명 |
|------|------|----------|------|
| id | UUID | PK | 고유 식별자 |
| image_id | UUID | FK → Image | 원본 이미지 |
| path | VARCHAR(512) | NOT NULL | 파일 경로 |
| num_phases | INTEGER | NOT NULL | 상 개수 |
| phase_labels | JSON | NOT NULL | 상 라벨 매핑 |
| denoise_sigma | FLOAT | NULLABLE | 적용된 노이즈 제거 시그마 |
| segmentation_method | VARCHAR(50) | NOT NULL | 분할 방법 |
| created_at | TIMESTAMP | NOT NULL | 생성 일시 |

```python
@dataclass
class ProcessedData:
    id: str
    image_id: str
    path: str
    num_phases: int
    phase_labels: Dict[int, str]  # {0: "pore", 1: "active", 2: "binder"}
    denoise_sigma: Optional[float] = None
    segmentation_method: str = "threshold"
    created_at: datetime = field(default_factory=datetime.now)
```

### 3.4 ModelConfig (모델 설정)

SliceGAN 모델 설정을 저장합니다.

| 필드 | 타입 | 제약조건 | 설명 |
|------|------|----------|------|
| id | UUID | PK | 고유 식별자 |
| name | VARCHAR(255) | NOT NULL | 설정 이름 |
| nz | INTEGER | NOT NULL | 잠재 벡터 차원 |
| ngf | INTEGER | NOT NULL | Generator 필터 수 |
| ndf | INTEGER | NOT NULL | Discriminator 필터 수 |
| nc | INTEGER | NOT NULL | 출력 채널 (상 개수) |
| image_size | INTEGER | NOT NULL | 출력 크기 |
| batch_size | INTEGER | NOT NULL | 배치 크기 |
| lr_g | FLOAT | NOT NULL | Generator 학습률 |
| lr_d | FLOAT | NOT NULL | Discriminator 학습률 |
| lambda_gp | FLOAT | NOT NULL | Gradient penalty 가중치 |
| n_critic | INTEGER | NOT NULL | Critic 업데이트 횟수 |
| epochs | INTEGER | NOT NULL | 학습 에폭 |
| created_at | TIMESTAMP | NOT NULL | 생성 일시 |

```python
@dataclass
class ModelConfig:
    id: str
    name: str
    nz: int = 64
    ngf: int = 64
    ndf: int = 64
    nc: int = 3
    image_size: int = 64
    batch_size: int = 8
    lr_g: float = 0.0001
    lr_d: float = 0.0004
    lambda_gp: float = 10.0
    n_critic: int = 5
    epochs: int = 100
    created_at: datetime = field(default_factory=datetime.now)
```

### 3.5 Model (학습된 모델)

학습된 SliceGAN 모델 정보를 저장합니다.

| 필드 | 타입 | 제약조건 | 설명 |
|------|------|----------|------|
| id | UUID | PK | 고유 식별자 |
| config_id | UUID | FK → ModelConfig | 모델 설정 |
| processed_data_id | UUID | FK → ProcessedData | 훈련 데이터 |
| checkpoint_path | VARCHAR(512) | NOT NULL | 체크포인트 경로 |
| best_loss | FLOAT | NULLABLE | 최저 손실값 |
| trained_epochs | INTEGER | NOT NULL | 학습 에폭 수 |
| trained_at | TIMESTAMP | NOT NULL | 학습 완료 일시 |
| status | VARCHAR(20) | NOT NULL | 상태 (training, completed, failed) |

```python
@dataclass
class Model:
    id: str
    config_id: str
    processed_data_id: str
    checkpoint_path: str
    best_loss: Optional[float] = None
    trained_epochs: int = 0
    trained_at: Optional[datetime] = None
    status: str = "pending"  # pending, training, completed, failed
```

### 3.6 TrainingLog (훈련 로그)

훈련 과정 로그를 저장합니다.

| 필드 | 타입 | 제약조건 | 설명 |
|------|------|----------|------|
| id | UUID | PK | 고유 식별자 |
| model_id | UUID | FK → Model | 소속 모델 |
| epoch | INTEGER | NOT NULL | 에폭 번호 |
| g_loss | FLOAT | NOT NULL | Generator 손실 |
| d_loss | FLOAT | NOT NULL | Discriminator 손실 |
| d_real | FLOAT | NULLABLE | Real 판별 점수 |
| d_fake | FLOAT | NULLABLE | Fake 판별 점수 |
| timestamp | TIMESTAMP | NOT NULL | 기록 시간 |

```python
@dataclass
class TrainingLog:
    id: str
    model_id: str
    epoch: int
    g_loss: float
    d_loss: float
    d_real: Optional[float] = None
    d_fake: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
```

### 3.7 Volume (3D 볼륨)

생성된 3D 복셀 볼륨 정보를 저장합니다.

| 필드 | 타입 | 제약조건 | 설명 |
|------|------|----------|------|
| id | UUID | PK | 고유 식별자 |
| model_id | UUID | FK → Model | 생성 모델 |
| path | VARCHAR(512) | NOT NULL | 파일 경로 (.npy) |
| shape | JSON | NOT NULL | 볼륨 형상 [D, H, W] |
| dtype | VARCHAR(20) | NOT NULL | 데이터 타입 |
| voxel_size | FLOAT | NULLABLE | 복셀 크기 (μm) |
| seed | INTEGER | NULLABLE | 생성 시드 |
| created_at | TIMESTAMP | NOT NULL | 생성 일시 |

```python
@dataclass
class Volume:
    id: str
    model_id: str
    path: str
    shape: Tuple[int, int, int]
    dtype: str = "uint8"
    voxel_size: Optional[float] = None
    seed: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)
```

### 3.8 Mesh (메시)

변환된 표면 메시 정보를 저장합니다.

| 필드 | 타입 | 제약조건 | 설명 |
|------|------|----------|------|
| id | UUID | PK | 고유 식별자 |
| volume_id | UUID | FK → Volume | 소스 볼륨 |
| phase_id | INTEGER | NOT NULL | 추출된 상 ID |
| path | VARCHAR(512) | NOT NULL | 파일 경로 |
| format | VARCHAR(10) | NOT NULL | 형식 (stl, obj, vtk) |
| num_vertices | INTEGER | NOT NULL | 정점 수 |
| num_faces | INTEGER | NOT NULL | 면 수 |
| is_watertight | BOOLEAN | NOT NULL | 수밀성 여부 |
| surface_area | FLOAT | NULLABLE | 표면적 |
| volume_value | FLOAT | NULLABLE | 부피 |
| created_at | TIMESTAMP | NOT NULL | 생성 일시 |

```python
@dataclass
class Mesh:
    id: str
    volume_id: str
    phase_id: int
    path: str
    format: str
    num_vertices: int
    num_faces: int
    is_watertight: bool
    surface_area: Optional[float] = None
    volume_value: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
```

### 3.9 Metrics (메트릭)

계산된 미세구조 메트릭을 저장합니다.

| 필드 | 타입 | 제약조건 | 설명 |
|------|------|----------|------|
| id | UUID | PK | 고유 식별자 |
| volume_id | UUID | FK → Volume | 소스 볼륨 |
| porosity | FLOAT | NOT NULL | 기공률 |
| active_material_fraction | FLOAT | NOT NULL | 활물질 분율 |
| specific_surface_area | FLOAT | NOT NULL | 비표면적 |
| tortuosity_x | FLOAT | NULLABLE | X방향 굴곡도 |
| tortuosity_y | FLOAT | NULLABLE | Y방향 굴곡도 |
| tortuosity_z | FLOAT | NULLABLE | Z방향 굴곡도 |
| connectivity_ratio | FLOAT | NULLABLE | 연결성 비율 |
| num_particles | INTEGER | NULLABLE | 입자 수 |
| psd_data | JSON | NULLABLE | 입자 크기 분포 데이터 |
| created_at | TIMESTAMP | NOT NULL | 계산 일시 |

```python
@dataclass
class Metrics:
    id: str
    volume_id: str
    porosity: float
    active_material_fraction: float
    specific_surface_area: float
    tortuosity_x: Optional[float] = None
    tortuosity_y: Optional[float] = None
    tortuosity_z: Optional[float] = None
    connectivity_ratio: Optional[float] = None
    num_particles: Optional[int] = None
    psd_data: Optional[Dict[str, List[float]]] = None
    created_at: datetime = field(default_factory=datetime.now)
```

### 3.10 Visualization (시각화)

생성된 시각화 결과물 정보를 저장합니다.

| 필드 | 타입 | 제약조건 | 설명 |
|------|------|----------|------|
| id | UUID | PK | 고유 식별자 |
| volume_id | UUID | FK → Volume | 소스 볼륨 |
| type | VARCHAR(50) | NOT NULL | 시각화 유형 |
| path | VARCHAR(512) | NOT NULL | 파일 경로 |
| settings | JSON | NULLABLE | 시각화 설정 |
| created_at | TIMESTAMP | NOT NULL | 생성 일시 |

```python
@dataclass
class Visualization:
    id: str
    volume_id: str
    type: str  # "slice", "orthogonal", "3d_render", "animation"
    path: str
    settings: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)
```

### 3.11 Simulation (시뮬레이션)

COMSOL 시뮬레이션 정보를 저장합니다.

| 필드 | 타입 | 제약조건 | 설명 |
|------|------|----------|------|
| id | UUID | PK | 고유 식별자 |
| mesh_id | UUID | FK → Mesh | 소스 메시 |
| tool | VARCHAR(50) | NOT NULL | 시뮬레이션 도구 |
| type | VARCHAR(50) | NOT NULL | 시뮬레이션 유형 |
| parameters | JSON | NOT NULL | 시뮬레이션 파라미터 |
| model_path | VARCHAR(512) | NULLABLE | COMSOL 모델 경로 |
| results_path | VARCHAR(512) | NULLABLE | 결과 파일 경로 |
| status | VARCHAR(20) | NOT NULL | 상태 |
| error_message | TEXT | NULLABLE | 에러 메시지 |
| started_at | TIMESTAMP | NULLABLE | 시작 일시 |
| completed_at | TIMESTAMP | NULLABLE | 완료 일시 |
| created_at | TIMESTAMP | NOT NULL | 생성 일시 |

```python
@dataclass
class Simulation:
    id: str
    mesh_id: str
    tool: str = "comsol"
    type: str = "electrochemistry"
    parameters: Dict[str, Any] = field(default_factory=dict)
    model_path: Optional[str] = None
    results_path: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
```

---

## 4. 관계 정의

### 4.1 관계 요약

| 관계 | 카디널리티 | 설명 |
|------|-----------|------|
| Project → Image | 1:N | 프로젝트는 여러 이미지를 포함 |
| Image → ProcessedData | 1:N | 이미지는 여러 전처리 결과를 가질 수 있음 |
| ModelConfig → Model | 1:N | 설정으로 여러 모델 학습 가능 |
| ProcessedData → Model | 1:N | 같은 데이터로 여러 모델 학습 |
| Model → TrainingLog | 1:N | 모델당 여러 학습 로그 |
| Model → Volume | 1:N | 모델로 여러 볼륨 생성 |
| Volume → Mesh | 1:N | 볼륨에서 여러 메시 추출 (상별) |
| Volume → Metrics | 1:1 | 볼륨당 하나의 메트릭 세트 |
| Volume → Visualization | 1:N | 볼륨에서 여러 시각화 생성 |
| Mesh → Simulation | 1:N | 메시로 여러 시뮬레이션 실행 |

### 4.2 외래 키 제약조건

```sql
-- 외래 키 정의
ALTER TABLE Image ADD CONSTRAINT fk_image_project
    FOREIGN KEY (project_id) REFERENCES Project(id) ON DELETE CASCADE;

ALTER TABLE ProcessedData ADD CONSTRAINT fk_processed_image
    FOREIGN KEY (image_id) REFERENCES Image(id) ON DELETE CASCADE;

ALTER TABLE Model ADD CONSTRAINT fk_model_config
    FOREIGN KEY (config_id) REFERENCES ModelConfig(id);

ALTER TABLE Model ADD CONSTRAINT fk_model_processed
    FOREIGN KEY (processed_data_id) REFERENCES ProcessedData(id);

ALTER TABLE TrainingLog ADD CONSTRAINT fk_log_model
    FOREIGN KEY (model_id) REFERENCES Model(id) ON DELETE CASCADE;

ALTER TABLE Volume ADD CONSTRAINT fk_volume_model
    FOREIGN KEY (model_id) REFERENCES Model(id);

ALTER TABLE Mesh ADD CONSTRAINT fk_mesh_volume
    FOREIGN KEY (volume_id) REFERENCES Volume(id) ON DELETE CASCADE;

ALTER TABLE Metrics ADD CONSTRAINT fk_metrics_volume
    FOREIGN KEY (volume_id) REFERENCES Volume(id) ON DELETE CASCADE;

ALTER TABLE Visualization ADD CONSTRAINT fk_viz_volume
    FOREIGN KEY (volume_id) REFERENCES Volume(id) ON DELETE CASCADE;

ALTER TABLE Simulation ADD CONSTRAINT fk_sim_mesh
    FOREIGN KEY (mesh_id) REFERENCES Mesh(id);
```

---

## 5. 인덱스 정의

### 5.1 기본 인덱스

```sql
-- 자주 조회되는 필드에 인덱스 생성
CREATE INDEX idx_image_project ON Image(project_id);
CREATE INDEX idx_image_created ON Image(created_at);

CREATE INDEX idx_processed_image ON ProcessedData(image_id);

CREATE INDEX idx_model_config ON Model(config_id);
CREATE INDEX idx_model_status ON Model(status);

CREATE INDEX idx_log_model ON TrainingLog(model_id);
CREATE INDEX idx_log_epoch ON TrainingLog(model_id, epoch);

CREATE INDEX idx_volume_model ON Volume(model_id);
CREATE INDEX idx_volume_created ON Volume(created_at);

CREATE INDEX idx_mesh_volume ON Mesh(volume_id);
CREATE INDEX idx_mesh_format ON Mesh(format);

CREATE INDEX idx_metrics_volume ON Metrics(volume_id);

CREATE INDEX idx_sim_mesh ON Simulation(mesh_id);
CREATE INDEX idx_sim_status ON Simulation(status);
```

---

## 6. 데이터 흐름 시퀀스

```
┌────────┐     ┌────────┐     ┌────────────┐     ┌────────┐     ┌────────┐
│Project │────▶│ Image  │────▶│ProcessedData│────▶│ Model  │────▶│ Volume │
└────────┘     └────────┘     └────────────┘     └────────┘     └────┬───┘
                                    │                   │             │
                                    │                   │             │
                                    ▼                   ▼             │
                              ┌───────────┐     ┌───────────┐        │
                              │ModelConfig│     │TrainingLog│        │
                              └───────────┘     └───────────┘        │
                                                                      │
                    ┌─────────────────────────────────────────────────┤
                    │                         │                       │
                    ▼                         ▼                       ▼
             ┌────────────┐           ┌───────────┐         ┌─────────────┐
             │    Mesh    │           │  Metrics  │         │Visualization│
             └─────┬──────┘           └───────────┘         └─────────────┘
                   │
                   ▼
            ┌────────────┐
            │ Simulation │
            └────────────┘
```

---

## 7. 데이터 무결성 규칙

### 7.1 비즈니스 규칙

1. **Volume 생성 규칙**
   - Model.status가 "completed"인 경우에만 Volume 생성 가능
   - Volume.shape는 [D, H, W] 형식이어야 함

2. **Mesh 생성 규칙**
   - Volume.phase_id는 해당 Volume의 num_phases 범위 내여야 함
   - is_watertight가 False인 경우 Simulation 생성 시 경고

3. **Simulation 생성 규칙**
   - Mesh.is_watertight가 True인 경우에만 COMSOL 시뮬레이션 권장
   - parameters에 필수 키 포함 확인

### 7.2 데이터 검증

```python
def validate_volume_shape(shape: Tuple[int, int, int]) -> bool:
    """볼륨 형상 검증"""
    if len(shape) != 3:
        return False
    return all(s > 0 and s <= 512 for s in shape)

def validate_metrics(metrics: Metrics) -> bool:
    """메트릭 값 범위 검증"""
    if not 0 <= metrics.porosity <= 1:
        return False
    if not 0 <= metrics.active_material_fraction <= 1:
        return False
    if metrics.specific_surface_area < 0:
        return False
    return True

def validate_simulation_parameters(params: Dict[str, Any], sim_type: str) -> bool:
    """시뮬레이션 파라미터 검증"""
    required_keys = {
        "electrochemistry": ["temperature", "c_rate"],
        "transport": ["diffusivity", "conductivity"],
    }
    return all(key in params for key in required_keys.get(sim_type, []))
```

---

## 8. 마이그레이션 고려사항

### 8.1 버전 관리

| 버전 | 변경사항 |
|------|----------|
| v1.0.0 | 초기 스키마 |
| v1.1.0 | Simulation 테이블 추가 예정 |
| v1.2.0 | Batch processing 지원 예정 |

### 8.2 데이터 마이그레이션 전략

1. **백업**: 마이그레이션 전 전체 데이터 백업
2. **스키마 변경**: 새 필드 추가 시 DEFAULT 값 설정
3. **데이터 변환**: 기존 데이터 새 형식으로 변환
4. **검증**: 마이그레이션 후 데이터 무결성 검증

---

*문서 끝*
